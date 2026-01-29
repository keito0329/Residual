"""
Pytorch Lightning Modules.
"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from src.models import BERT4Rec, LightSASRecAnalyze, SASRecAnalyze  # Added SASRecAnalyze
from pytorch_lightning.trainer.states import RunningStage

class SeqRecBase(pl.LightningModule):

    def __init__(self, model, lr=1e-3, padding_idx=0,
                 predict_top_k=10, filter_seen=True):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):
        
            preds, scores = self.make_prediction(batch, batch_idx=batch_idx)

            scores = scores.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            user_ids = batch['user_id'].detach().cpu().numpy()
            
            if 'target_ids' in batch:
                target_ids = batch['target_ids'].detach().cpu().numpy()
                return {'preds': preds,
                        'scores': scores,
                        'user_ids': user_ids,
                        'target_ids': target_ids}
            
            return {'preds': preds,
                    'scores': scores,
                    'user_ids': user_ids}

    def validation_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch['target'], preds)

        self.log("val_ndcg", metrics['ndcg'], prog_bar=True)
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True)
        self.log("val_mrr", metrics['mrr'], prog_bar=True)

    def make_prediction(self, batch, batch_idx=None):
        
        outputs = self.prediction_output(batch, batch_idx=batch_idx)

        input_ids = batch['input_ids']
        rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1
        position_offset = int(getattr(self, "position_offset", 0))  # Added (Changed) by Author
        last_item_idx = torch.clamp(last_item_idx - position_offset, min=0)  # Added (Changed) by Author
        preds = outputs[rows_ids, last_item_idx, :]

        scores, preds = torch.sort(preds, descending=True)

        if self.filter_seen:
            seen_items = batch['seen_ids'] 
            preds, scores = self.filter_seen_items(preds, scores, seen_items)
        else:
            scores = scores[:, :self.predict_top_k]
            preds = preds[:, :self.predict_top_k]

        return preds, scores
    
    def filter_seen_items(self, preds, scores, seen_items):

        max_len = seen_items.size(1)
        scores = scores[:, :self.predict_top_k + max_len]
        preds = preds[:, :self.predict_top_k + max_len]

        final_preds, final_scores = [], []
        for i in range(preds.size(0)):
            not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
            pred = preds[i, not_seen_indexes][:self.predict_top_k]
            score = scores[i, not_seen_indexes][:self.predict_top_k]
            final_preds.append(pred)
            final_scores.append(score)

        final_preds = torch.vstack(final_preds)
        final_scores = torch.vstack(final_scores)

        return final_preds, final_scores

    def compute_val_metrics(self, targets, preds):

        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}


class SeqRec(SeqRecBase):

    def __init__(self, model, lr=1e-3, padding_idx=0,
                 predict_top_k=10, filter_seen=True,
                 use_lmp=False, lmp_h=0, lmp_lambda=None, lmp_decay=0.5,
                 causal_mask_at_inference=False, save_analysis_npz=True):
        # Added for LMP: keep backward compat defaults
        super().__init__(model, lr=lr, padding_idx=padding_idx,
                         predict_top_k=predict_top_k, filter_seen=filter_seen)
        # Added for LMP
        self.use_lmp = use_lmp
        self.lmp_h = lmp_h
        self.lmp_lambda = lmp_lambda
        self.lmp_decay = lmp_decay
        self.causal_mask_at_inference = causal_mask_at_inference
        self.save_analysis_npz = save_analysis_npz

    def _build_causal_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = attention_mask.size(1)
        causal = torch.tril(
            torch.ones((seq_len, seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
        )
        return attention_mask[:, None, :] * causal

    def training_step(self, batch, batch_idx):

        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, outputs, batch):

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))

        # Added for LMP: optional multi-step CE
        if self.use_lmp and self.lmp_h > 0:
            B, L, _ = outputs.shape
            labels = batch['labels']

            # decide per-step weights
            if self.lmp_lambda is not None:
                weights = self.lmp_lambda
            else:
                weights = [self.lmp_decay ** h for h in range(1, self.lmp_h + 1)]

            for h, w in enumerate(weights, start=1):
                if h >= L:
                    break
                shifted_logits = outputs[:, :-h, :]           # predict future h steps ahead
                shifted_labels = labels[:, h:]               # corresponding future labels
                step_loss = loss_fct(
                    shifted_logits.contiguous().view(-1, shifted_logits.size(-1)),
                    shifted_labels.contiguous().view(-1)
                )
                loss = loss + w * step_loss

        return loss
    def prediction_output(self, batch, batch_idx=None):

        stage = getattr(getattr(self, "trainer", None), "state", None)
        stage_val = getattr(stage, "stage", None)
        is_predict = stage_val == RunningStage.PREDICTING or str(stage_val).lower().endswith("predicting")

        analyzable_classes = (LightSASRecAnalyze, SASRecAnalyze)
        want_save_analysis = bool(getattr(self, "save_analysis_npz", True))
        want_analysis = isinstance(self.model, analyzable_classes) and is_predict and want_save_analysis

        if isinstance(self.model, analyzable_classes):
            output = self.model(
                batch['input_ids'],
                batch['attention_mask'],
                return_mixing=want_analysis,
                save_analysis=want_analysis,
                apply_residual_scale=is_predict,
                analysis_batch_idx=batch_idx if batch_idx is not None else getattr(self, "global_step", 0),
            )
        elif isinstance(self.model, BERT4Rec):
            want_norms = bool(getattr(self.model, "save_norms", False)) and is_predict and want_save_analysis
            residual_scale = (
                float(getattr(self.model, "residual_scale", 1.0)) if is_predict else 1.0
            )  # Added by Author
            attention_mask = batch['attention_mask']
            apply_causal = (
                is_predict
                and self.causal_mask_at_inference
                and bool(getattr(self, "_force_causal_infer", False))
            )
            if apply_causal:
                attention_mask = self._build_causal_attention_mask(attention_mask)
            output = self.model(
                batch['input_ids'],
                attention_mask,
                return_norms=want_norms,
                save_analysis=want_norms,
                analysis_batch_idx=batch_idx if batch_idx is not None else getattr(self, "global_step", 0),
                residual_scale=residual_scale,  # Added by Author
            )
        else:
            output = self.model(
                batch['input_ids'],
                batch['attention_mask'],
            )

        # LightSASRecAnalyze returns (logits, analysis) when return_mixing=True
        if isinstance(output, tuple):
            output, _ = output

        return output
    
    

    # def prediction_output(self, batch):
    #     if isinstance(self.model, LightSASRecAnalyze):
    #         # Call without return_analysis/save_analysis here
    #         outputs = self.model(batch['input_ids'], batch['attention_mask'])
    #     else:
    #         outputs = self.model(
    #             batch['input_ids'],
    #             batch['attention_mask'],
    #             return_analysis=not self.training,
    #             save_analysis=not self.training,
    #         )

    #     if isinstance(outputs, tuple):
    #         logits, analysis = outputs
    #         return logits
    #     return outputs
