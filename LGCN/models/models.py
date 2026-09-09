
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base_cmn import BaseCMN
from modules.report_fields import view_group

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as e:
    AutoModel = None
    AutoTokenizer = None


DEFAULT_VIEW_POSITION_DICT = {
    "PA": 0,
    "LATERAL": 1,
    "AP": 2,
    "LL": 3,
    "unk": 4,
    "LAO": 5,
    "RAO": 6,
    "AP AXIAL": 7,
    "SWIMMERS": 8,
    "PA LLD": 9,
    "AP LLD": 10,
    "XTABLE LATERAL": 11,
    "AP RLD": 12,
    "PA RLD": 13,
    "LPO": 14,
    "prior": 15,
}


class LightGraphConv(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        # adj: [B, L, L], non-negative edge weights.
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.bmm(adj / deg, x)
        return self.norm(x + self.drop(self.proj(neigh)))


class GraphMaskedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key_value, temporal_adj):
        # temporal_adj: [B, Lq, Lk], positive/1 means allowed.
        bsz, lq, lk = temporal_adj.shape
        # MultiheadAttention bool attn_mask: True means blocked.
        blocked = ~(temporal_adj > 0)
        # ensure every query has at least one allowed key
        no_edge = blocked.all(dim=-1)
        if no_edge.any():
            blocked = blocked.clone()
            blocked[no_edge, 0] = False
        mask = blocked.unsqueeze(1).expand(bsz, self.attn.num_heads, lq, lk)
        mask = mask.reshape(bsz * self.attn.num_heads, lq, lk)
        out, _ = self.attn(query=query, key=key_value, value=key_value, attn_mask=mask, need_weights=False)
        return self.norm(query + self.drop(out))


class ViewAwareGraphTemporalDifferenceFusion(nn.Module):
    """
    Simplified graph image branch:
    - builds current/prior patch graphs on pooled pure visual tokens
    - adds view/time embeddings after pooling
    - performs one lightweight GCN for current/prior
    - builds view-aware temporal graph
    - graph-masked cross attention from current to prior
    - difference/product fusion
    """
    def __init__(self, dim, num_heads=8, visual_topk=4, temporal_topk=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.visual_topk = int(visual_topk)
        self.temporal_topk = int(temporal_topk)
        self.cur_gcn = LightGraphConv(dim, dropout)
        self.pri_gcn = LightGraphConv(dim, dropout)
        self.cross = GraphMaskedCrossAttention(dim, num_heads, dropout)
        self.diff_fuse = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    @torch.no_grad()
    def _build_unimodal_adj(self, x, topk):
        bsz, length, _ = x.shape
        x_n = F.normalize(x, dim=-1)
        sim = torch.bmm(x_n, x_n.transpose(1, 2)).clamp_min(0.0)
        eye = torch.eye(length, device=x.device).unsqueeze(0).expand(bsz, -1, -1)
        adj = eye.clone()
        if topk > 0 and length > 1:
            k = min(topk + 1, length)
            idx = sim.topk(k=k, dim=-1).indices
            top_mask = torch.zeros_like(sim)
            top_mask.scatter_(dim=-1, index=idx, value=1.0)
            adj = torch.maximum(adj, sim * top_mask)
        return adj.detach()

    def _same_index_allowed(self, cur_views, pri_views, device):
        vals = []
        for cv, pv in zip(cur_views, pri_views):
            cg, pg = view_group(cv), view_group(pv)
            # AP/PA are both frontal and can use same-index; lateral only with lateral.
            vals.append((cg == "frontal" and pg == "frontal") or (cg == "lateral" and pg == "lateral"))
        return torch.tensor(vals, dtype=torch.bool, device=device)

    @torch.no_grad()
    def _build_temporal_adj(self, cur_graph_feat, pri_graph_feat, cur_views, pri_views):
        bsz, lc, _ = cur_graph_feat.shape
        lp = pri_graph_feat.size(1)
        device = cur_graph_feat.device
        adj = torch.zeros(bsz, lc, lp, device=device)

        # 1) same-index edge only for view-compatible pairs
        m = min(lc, lp)
        if m > 0 and cur_views is not None and pri_views is not None:
            same_mask = self._same_index_allowed(cur_views, pri_views, device)
            valid_b = torch.nonzero(same_mask, as_tuple=False).flatten()
            if valid_b.numel() > 0:
                ar = torch.arange(m, device=device)
                adj[valid_b[:, None], ar[None, :], ar[None, :]] = 1.0

        # 2) top-k cross-similarity edges always used
        if self.temporal_topk > 0 and lp > 0:
            sim = torch.bmm(
                F.normalize(cur_graph_feat, dim=-1),
                F.normalize(pri_graph_feat, dim=-1).transpose(1, 2)
            ).clamp_min(0.0)
            k = min(self.temporal_topk, lp)
            idx = sim.topk(k=k, dim=-1).indices
            top_mask = torch.zeros_like(sim)
            top_mask.scatter_(dim=-1, index=idx, value=1.0)
            adj = torch.maximum(adj, sim * top_mask)

        return adj.detach()

    def forward(self, cur_graph_feat, cur_fuse_feat, pri_graph_feat=None, pri_fuse_feat=None,
                has_prior=None, cur_views=None, pri_views=None):
        out = self.out_norm(cur_fuse_feat)
        if pri_graph_feat is None or pri_fuse_feat is None or has_prior is None or not has_prior.any():
            return out

        has_idx = torch.nonzero(has_prior, as_tuple=False).flatten()
        cur_g0 = cur_graph_feat.index_select(0, has_idx)
        cur_f0 = cur_fuse_feat.index_select(0, has_idx)
        pri_g0 = pri_graph_feat.index_select(0, has_idx)
        pri_f0 = pri_fuse_feat.index_select(0, has_idx)

        cur_views_has = [cur_views[int(i)] for i in has_idx.tolist()] if cur_views is not None else None
        pri_views_has = [pri_views[int(i)] for i in has_idx.tolist()] if pri_views is not None else None

        cur_adj = self._build_unimodal_adj(cur_g0, self.visual_topk)
        pri_adj = self._build_unimodal_adj(pri_g0, self.visual_topk)

        cur_g = self.cur_gcn(cur_f0, cur_adj)
        pri_g = self.pri_gcn(pri_f0, pri_adj)

        temporal_adj = self._build_temporal_adj(cur_g0, pri_g0, cur_views_has, pri_views_has)
        prior_aware = self.cross(cur_g, pri_g, temporal_adj)

        delta = self.diff_fuse(torch.cat([
            cur_g,
            prior_aware,
            cur_g - prior_aware,
            cur_g * prior_aware,
        ], dim=-1))
        fused = self.out_norm(cur_g + delta)

        out = out.clone()
        out.index_copy_(0, has_idx, fused)
        return out


class GsiTThreeSourceFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.fwd = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.bwd = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim * 2, dim)
        self.intra = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.num_heads = num_heads

    def _mask(self, lv, li, lp, mode, device):
        total = lv + li + lp
        allowed = torch.zeros(total, total, dtype=torch.bool, device=device)
        sv = (0, lv); si = (lv, lv + li); sp = (lv + li, total)
        groups = [sv, si, sp]
        if mode == "forward":
            pairs = [(0, 1), (1, 2), (2, 0)]  # V<-I, I<-P, P<-V
        elif mode == "backward":
            pairs = [(0, 2), (2, 1), (1, 0)]  # V<-P, P<-I, I<-V
        elif mode == "intra":
            pairs = [(0, 0), (1, 1), (2, 2)]
        else:
            raise ValueError(mode)
        for dst, src in pairs:
            ds, de = groups[dst]; ss, se = groups[src]
            allowed[ds:de, ss:se] = True
        return ~allowed  # bool attn_mask; True means blocked

    def forward(self, v, i, p, i_mask=None, p_mask=None):
        bsz, lv, dim = v.shape
        li = i.size(1)
        lp = p.size(1)
        x = torch.cat([v, i, p], dim=1)

        v_mask = torch.ones(bsz, lv, dtype=torch.bool, device=v.device)
        if i_mask is None:
            i_mask = torch.ones(bsz, li, dtype=torch.bool, device=v.device)
        if p_mask is None:
            p_mask = torch.ones(bsz, lp, dtype=torch.bool, device=v.device)
        token_mask = torch.cat([v_mask, i_mask.bool(), p_mask.bool()], dim=1)
        key_padding_mask = ~token_mask

        fwd_mask = self._mask(lv, li, lp, "forward", v.device)
        bwd_mask = self._mask(lv, li, lp, "backward", v.device)
        intra_mask = self._mask(lv, li, lp, "intra", v.device)

        y1, _ = self.fwd(x, x, x, attn_mask=fwd_mask, key_padding_mask=key_padding_mask, need_weights=False)
        y2, _ = self.bwd(x, x, x, attn_mask=bwd_mask, key_padding_mask=key_padding_mask, need_weights=False)
        y = self.norm1(x + self.drop(self.proj(torch.cat([y1, y2], dim=-1))))
        y3, _ = self.intra(y, y, y, attn_mask=intra_mask, key_padding_mask=key_padding_mask, need_weights=False)
        y = self.norm2(y + self.drop(y3))

        fv = y[:, :lv]
        fi = y[:, lv:lv+li]
        fp = y[:, lv+li:]
        return fv, fi, fp


class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for RadDINO/CXR-BERT encoders.")
        self.args = args
        self.tokenizer = tokenizer
        self.d_model = args.d_model
        self.max_visual_tokens = int(getattr(args, "ours_max_visual_tokens", 49))
        self.decoder_text_mode = getattr(args, "decoder_text_mode", "indication")

        # Local frozen encoders
        self.image_encoder = AutoModel.from_pretrained(
            args.rad_dino_path, trust_remote_code=True, local_files_only=True
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            args.cxr_bert_path, trust_remote_code=True, local_files_only=True
        )
        self.text_encoder = AutoModel.from_pretrained(
            args.cxr_bert_path, trust_remote_code=True, local_files_only=True
        )
        special = {"additional_special_tokens": ["[INDICATION]", "[PRIOR_REPORT]", "[FINDINGS]", "[NHI]", "[NHPR]"]}
        self.text_tokenizer.add_special_tokens(special)
        self.text_encoder.resize_token_embeddings(len(self.text_tokenizer))

        if getattr(args, "freeze_image_encoder", True):
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if getattr(args, "freeze_text_encoder", True):
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        img_dim = getattr(self.image_encoder.config, "hidden_size", None) or getattr(self.image_encoder.config, "hidden_dim", None)
        txt_dim = getattr(self.text_encoder.config, "hidden_size", None) or getattr(self.text_encoder.config, "hidden_dim", None)
        if img_dim is None or txt_dim is None:
            raise ValueError("Cannot infer hidden size from RadDINO/CXR-BERT config.")

        self.image_projection = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Linear(img_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(txt_dim),
            nn.Linear(txt_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.vp2id = DEFAULT_VIEW_POSITION_DICT
        self.view_embed = nn.Embedding(len(self.vp2id), self.d_model)
        self.time_embed = nn.Embedding(2, self.d_model)  # 0=current, 1=prior

        self.visual_fusion = ViewAwareGraphTemporalDifferenceFusion(
            self.d_model, args.num_heads,
            visual_topk=getattr(args, "visual_graph_topk", 4),
            temporal_topk=getattr(args, "temporal_graph_topk", 4),
            dropout=args.dropout,
        )
        self.gsit = GsiTThreeSourceFusion(self.d_model, args.num_heads, args.dropout)

        # Clean Transformer decoder from R2GenCMN, CMN memory disabled in modules/base_cmn.py.
        # The memory produced by our fusion module already has dimension d_model.
        self.args.d_vf = self.d_model
        args.d_vf = self.d_model
        self.encoder_decoder = BaseCMN(args, tokenizer)
        # compatibility with optimizer builder: use image_encoder as visual_extractor group
        self.visual_extractor = self.image_encoder

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _view_ids(self, views, device):
        ids = []
        for v in views:
            v = str(v).upper() if v is not None else "unk"
            ids.append(self.vp2id.get(v, self.vp2id["unk"]))
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _pool_tokens(self, x):
        # x: [B, L, D] -> [B, max_visual_tokens, D]
        if x.size(1) == self.max_visual_tokens:
            return x
        if x.size(1) < self.max_visual_tokens:
            pad = x.new_zeros(x.size(0), self.max_visual_tokens - x.size(1), x.size(2))
            return torch.cat([x, pad], dim=1)
        x_t = x.transpose(1, 2)
        x_t = F.adaptive_avg_pool1d(x_t, self.max_visual_tokens)
        return x_t.transpose(1, 2)

    def _encode_images(self, imgs):
        grad_enabled = any(p.requires_grad for p in self.image_encoder.parameters())
        with torch.set_grad_enabled(grad_enabled and self.training):
            out = self.image_encoder(pixel_values=imgs)
            tokens = out.last_hidden_state
        # remove CLS token if present and remaining tokens look patch-like
        if tokens.size(1) > 1:
            n = tokens.size(1) - 1
            s = int(math.sqrt(n))
            if s * s == n:
                tokens = tokens[:, 1:]
        tokens = self.image_projection(tokens)
        tokens = self._pool_tokens(tokens)
        return tokens

    def _add_view_time(self, tokens, views, time_id):
        device = tokens.device
        view_ids = self._view_ids(views, device)
        vemb = self.view_embed(view_ids).unsqueeze(1)
        temb = self.time_embed(torch.full((tokens.size(0),), time_id, dtype=torch.long, device=device)).unsqueeze(1)
        return tokens + vemb + temb

    def _encode_texts(self, texts, device, max_length):
        batch = self.text_tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", return_token_type_ids=False
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        grad_enabled = any(p.requires_grad for p in self.text_encoder.parameters())
        with torch.set_grad_enabled(grad_enabled and self.training):
            out = self.text_encoder(**batch).last_hidden_state
        return self.text_projection(out), batch["attention_mask"].bool()

    def _build_memory(self, images, extra):
        # images: [B, 2, C, H, W]
        if images.dim() == 4:
            cur_img = images
            pri_img = images
            has_prior = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        else:
            cur_img = images[:, 0]
            pri_img = images[:, 1]
            has_prior = extra.get("has_prior", torch.ones(images.size(0), dtype=torch.bool, device=images.device)).to(images.device)

        cur_graph = self._encode_images(cur_img)
        pri_graph = self._encode_images(pri_img)

        cur_views = extra.get("current_view", ["unk"] * images.size(0))
        pri_views = extra.get("prior_view", ["unk"] * images.size(0))

        cur_fuse = self._add_view_time(cur_graph, cur_views, 0)
        pri_fuse = self._add_view_time(pri_graph, pri_views, 1)

        visual_tokens = self.visual_fusion(
            cur_graph, cur_fuse, pri_graph, pri_fuse,
            has_prior=has_prior,
            cur_views=cur_views,
            pri_views=pri_views,
        )

        indication_len = int(getattr(self.args, "indication_max_length", 48))
        prior_len = int(getattr(self.args, "prior_report_max_length", 160))
        indication_embed, indication_mask = self._encode_texts(
            extra.get("indication_texts", ["[NHI]"] * images.size(0)),
            images.device,
            indication_len,
        )
        prior_embed, prior_mask = self._encode_texts(
            extra.get("prior_report_texts", ["[NHPR]"] * images.size(0)),
            images.device,
            prior_len,
        )

        fused_v, fused_i, fused_p = self.gsit(
            visual_tokens, indication_embed, prior_embed,
            i_mask=indication_mask,
            p_mask=prior_mask,
        )

        visual_mask = torch.ones(fused_v.size()[:2], dtype=torch.bool, device=images.device)
        if self.decoder_text_mode == "none":
            memory = fused_v
            memory_mask = visual_mask
        elif self.decoder_text_mode == "all":
            memory = torch.cat([fused_v, fused_i, fused_p], dim=1)
            memory_mask = torch.cat([visual_mask, indication_mask, prior_mask], dim=1)
        else:
            # default: visual + indication; prior report influences via GsiT but does not directly enter decoder.
            memory = torch.cat([fused_v, fused_i], dim=1)
            memory_mask = torch.cat([visual_mask, indication_mask], dim=1)

        assert memory.size(1) == memory_mask.size(1), (
            f"[VLPGF _build_memory] memory length {memory.size(1)} != "
            f"memory_mask length {memory_mask.size(1)} | mode={self.decoder_text_mode}"
        )

        # BaseCMN expects masks as 1/0 Long [B, L]; it will unsqueeze internally.
        return memory, memory_mask.long()

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}, extra=None):
        if extra is None:
            extra = {}
        memory, memory_mask = self._build_memory(images, extra)
        fc_feats = memory.new_zeros(memory.size(0), 1, self.d_model)
        if mode == 'train':
            return self.encoder_decoder(fc_feats, memory, targets, att_masks=memory_mask, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(fc_feats, memory, att_masks=memory_mask, mode='sample', update_opts=update_opts)
        else:
            raise ValueError(mode)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}, extra=None):
        return self.forward_mimic_cxr(images, targets=targets, mode=mode, update_opts=update_opts, extra=extra)

    def forward(self, images, targets=None, mode='train', update_opts={}, extra=None):
        if self.args.dataset_name == 'iu_xray':
            return self.forward_iu_xray(images, targets, mode, update_opts, extra)
        return self.forward_mimic_cxr(images, targets, mode, update_opts, extra)
