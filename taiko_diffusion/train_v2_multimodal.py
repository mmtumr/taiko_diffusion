from __future__ import annotations

import argparse
import csv
import json
import random
import itertools
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

from taiko_diffusion.models.v2_multimodal import V2MultimodalEncoder, V2MultimodalLocalAttentionEncoder, V2TechniqueHeadEncoder


LABELS = ["v2_main", "v2_stamina", "v2_handspeed", "v2_burst", "v2_complex", "v2_rhythm"]
AXIS_INDEX = {"stamina": 1, "handspeed": 2, "burst": 3, "complex": 4, "rhythm": 5}
RANK_WEIGHTS = {"stamina": 0.20, "handspeed": 0.05, "burst": 0.10, "complex": 0.08, "rhythm": 0.10}


class DatasetV2(Dataset):
    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray, target_map=None):
        with (root / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.mean, self.std, self.target_map = mean, std, target_map

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; data = np.load(row["npz_path"])
        y_raw = (self.target_map[row["sample_id"]] if self.target_map is not None else data["y"]).astype(np.float32); y = (y_raw - self.mean) / self.std
        return {"chart": torch.from_numpy(data["chart"]), "hand": torch.from_numpy(data["hand"]),
                "audio": torch.from_numpy(data["audio"]), "y": torch.from_numpy(y),
                "y_raw": torch.from_numpy(y_raw), "sample_id": row["sample_id"]}


class PairDataset(Dataset):
    def __init__(self, cache: Path, pair_csv: Path):
        rows = []
        for split in ("train", "val", "test"):
            with (cache / f"{split}.csv").open("r", encoding="utf-8-sig", newline="") as file:
                rows.extend(csv.DictReader(file))
        self.paths = {int(row["sample_id"].rsplit("_r", 1)[1]): row["npz_path"] for row in rows}
        with pair_csv.open("r", encoding="utf-8-sig", newline="") as file:
            self.rows = list(csv.DictReader(file))

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        high = np.load(self.paths[int(row["higher_rating_index"])])
        low = np.load(self.paths[int(row["lower_rating_index"])])
        result = {"axis": torch.tensor(AXIS_INDEX[row["axis"]], dtype=torch.long),
                  "weight": torch.tensor(RANK_WEIGHTS[row["axis"]] * min(max(float(row["confidence"]), 1.0), 3.0), dtype=torch.float32)}
        for prefix, data in (("high", high), ("low", low)):
            for name in ("chart", "hand", "audio"): result[f"{prefix}_{name}"] = torch.from_numpy(data[name])
        return result


def evaluate(model, loader, device, mean, std, no_audio=False, no_hand=False):
    model.eval(); pred, true = [], []
    with torch.no_grad():
        for b in loader:
            chart=b["chart"].to(device); hand=b["hand"].to(device); audio=b["audio"].to(device)
            if no_hand: hand=torch.zeros_like(hand)
            if no_audio: audio=torch.zeros_like(audio)
            p = model(chart, hand, audio)
            pred.append(p.cpu().numpy() * std + mean); true.append(b["y_raw"].numpy())
    p, y = np.concatenate(pred), np.concatenate(true); result = {}
    for i, name in enumerate(LABELS):
        error = p[:, i] - y[:, i]
        result[name] = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                        "pearson": float(pearsonr(p[:, i], y[:, i]).statistic),
                        "spearman": float(spearmanr(p[:, i], y[:, i]).statistic),
                        "r2": float(1 - np.sum(error ** 2) / np.sum((y[:, i] - y[:, i].mean()) ** 2))}
    return result


def pair_accuracy(model, dataset, device):
    loader=DataLoader(dataset,batch_size=24,shuffle=False);correct={axis:[0,0] for axis in AXIS_INDEX}
    model.eval()
    with torch.no_grad():
        for b in loader:
            high=model(b["high_chart"].to(device),b["high_hand"].to(device),b["high_audio"].to(device))
            low=model(b["low_chart"].to(device),b["low_hand"].to(device),b["low_audio"].to(device))
            for row,axis_index in enumerate(b["axis"].tolist()):
                axis=next(name for name,index in AXIS_INDEX.items() if index==axis_index);correct[axis][1]+=1;correct[axis][0]+=int(high[row,axis_index]>low[row,axis_index])
    return {axis:{"correct":value[0],"total":value[1],"accuracy":value[0]/max(value[1],1)} for axis,value in correct.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cache", type=Path, default=Path("data/cache/encoder_v2_multimodal"));
    parser.add_argument("--output", type=Path, default=Path("checkpoints/encoder_v2_multimodal")); parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--no-audio", action="store_true"); parser.add_argument("--no-hand", action="store_true")
    parser.add_argument("--physical-loss", type=float, default=0.0)
    parser.add_argument("--local-attention", action="store_true")
    parser.add_argument("--technique-heads", action="store_true")
    parser.add_argument("--pair-dir", type=Path, default=None)
    parser.add_argument("--targets", type=Path, default=None)
    args = parser.parse_args(); random.seed(20260619); np.random.seed(20260619); torch.manual_seed(20260619)
    stats = json.loads((args.cache / "stats.json").read_text(encoding="utf-8")); mean=np.asarray(stats["mean"],np.float32);std=np.asarray(stats["std"],np.float32);target_map=None
    if args.targets:
        import pandas as pd
        target_frame=pd.read_csv(args.targets);custom_columns=[f"custom_{name.removeprefix('v2_')}" for name in LABELS];target_map={str(row.sample_id):np.asarray([getattr(row,column) for column in custom_columns],np.float32) for row in target_frame.itertuples()}
        with (args.cache/"train.csv").open("r",encoding="utf-8-sig",newline="") as file: train_ids=[row["sample_id"] for row in csv.DictReader(file)]
        train_targets=np.stack([target_map[sample_id] for sample_id in train_ids]);mean=train_targets.mean(0);std=np.maximum(train_targets.std(0),1e-6)
    loaders={s:DataLoader(DatasetV2(args.cache,s,mean,std,target_map),batch_size=24,shuffle=s=="train",num_workers=0) for s in ("train","val","test")}
    pair_train = PairDataset(args.cache, args.pair_dir / "consensus_pairs_train.csv") if args.pair_dir else None
    pair_loader = DataLoader(pair_train,batch_size=16,shuffle=True,num_workers=0) if pair_train else None
    sample=np.load(loaders["train"].dataset.rows[0]["npz_path"]);hand_channels=int(sample["hand"].shape[0]);audio_channels=int(sample["audio"].shape[0]);chart_channels=int(sample["chart"].shape[0])
    model_class=V2TechniqueHeadEncoder if args.technique_heads else V2MultimodalLocalAttentionEncoder if args.local_attention else V2MultimodalEncoder
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=model_class(chart_channels,hand_channels,audio_channels).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1.5e-4)
    args.output.mkdir(parents=True,exist_ok=True);best=float("inf");patience=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();total=count=0;pair_iterator=itertools.cycle(pair_loader) if pair_loader else None
        for b in loaders["train"]:
            y=b["y"].to(device); teacher=y[:, :1] + torch.randn_like(y[:, :1])*0.08 if random.random()<0.5 else None
            chart=b["chart"].to(device); hand=b["hand"].to(device); audio=b["audio"].to(device)
            if args.no_hand: hand=torch.zeros_like(hand)
            if args.no_audio: audio=torch.zeros_like(audio)
            pred=model(chart,hand,audio,teacher)
            loss=torch.nn.functional.smooth_l1_loss(pred,y)
            if args.physical_loss > 0 and len(y) > 1:
                # The cache stores average-pooled channels first and max-pooled
                # channels second. Channels 24/25 are short/long note density.
                partner=torch.roll(torch.arange(len(y),device=device),1)
                proxies=[chart[:,25].mean(1),hand[:,:12].mean(dim=(1,2)),chart[:,51+24].amax(1)]
                for target_index,proxy in zip((1,2,3),proxies):
                    delta=proxy-proxy[partner]; pred_delta=pred[:,target_index]-pred[partner,target_index]
                    mask=delta.abs()>0.04
                    if mask.any(): loss=loss+args.physical_loss*torch.relu(0.05-torch.sign(delta[mask])*pred_delta[mask]).mean()
            if pair_iterator is not None:
                pair=next(pair_iterator); high=model(pair["high_chart"].to(device),pair["high_hand"].to(device),pair["high_audio"].to(device));low=model(pair["low_chart"].to(device),pair["low_hand"].to(device),pair["low_audio"].to(device));axis=pair["axis"].to(device);difference=(high-low).gather(1,axis[:,None]).squeeze(1);loss=loss+(pair["weight"].to(device)*torch.relu(0.15-difference)).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss)*len(y);count+=len(y)
        val=evaluate(model,loaders["val"],device,mean,std,args.no_audio,args.no_hand);val_loss=sum(v["mae"]/std[i] for i,v in enumerate(val.values()))/6;history.append({"epoch":epoch,"train":total/count,"val":val_loss})
        print(json.dumps(history[-1]),flush=True)
        if val_loss < best-0.002:
            best=val_loss;patience=0;torch.save({"model":model.state_dict(),"epoch":epoch,"mean":mean,"std":std,"local_attention":args.local_attention,"technique_heads":args.technique_heads,"physical_loss":args.physical_loss,"channels":{"chart":chart_channels,"hand":hand_channels,"audio":audio_channels}},args.output/"best.pt")
        else:
            patience+=1
            if patience>=10:break
    checkpoint=torch.load(args.output/"best.pt",map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"])
    report={"best_epoch":checkpoint["epoch"],"no_audio":args.no_audio,"no_hand":args.no_hand,"val":evaluate(model,loaders["val"],device,mean,std,args.no_audio,args.no_hand),"test":evaluate(model,loaders["test"],device,mean,std,args.no_audio,args.no_hand),"history":history}
    if args.pair_dir:
        report["pair_val"]=pair_accuracy(model,PairDataset(args.cache,args.pair_dir/"consensus_pairs_val.csv"),device)
        report["pair_test"]=pair_accuracy(model,PairDataset(args.cache,args.pair_dir/"consensus_pairs_test.csv"),device)
    (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report["test"],ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
