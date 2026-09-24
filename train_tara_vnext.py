"""Run a bounded TARA vNext pretraining experiment.

This is the bulk-training entry point: preflight, BPE, Transformer training,
validation, checkpointing and a deterministic sample are one command.
"""
import argparse
import random
from pathlib import Path

import torch

from src.text_dataset import load_dataset_text, list_datasets
from src.tokenizer import BPETokenizer
from src.torch_language_model import FastTinyLanguageModel


def windows(ids, context):
    if len(ids) <= context:
        raise ValueError("tokenized dataset is too short for context")
    return [(ids[i:i+context], ids[i+1:i+context+1]) for i in range(len(ids)-context)]


def batch(data, size, rng, device):
    chosen=[data[rng.randrange(len(data))] for _ in range(size)]
    return (torch.tensor([x for x,_ in chosen],dtype=torch.long,device=device),
            torch.tensor([y for _,y in chosen],dtype=torch.long,device=device))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=list_datasets(),default="tinystories")
    p.add_argument("--train-chars",type=int,default=262144)
    p.add_argument("--validation-chars",type=int,default=65536)
    p.add_argument("--vocab-size",type=int,default=4096)
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--context",type=int,default=256)
    p.add_argument("--embedding",type=int,default=256)
    p.add_argument("--ff-dim",type=int,default=1024)
    p.add_argument("--layers",type=int,default=6)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default=None)
    p.add_argument("--checkpoint",default="checkpoints/tara_vnext_tinystories.pt")
    args=p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_text=load_dataset_text(args.dataset,args.train_chars,"train")
    val_text=load_dataset_text(args.dataset,args.validation_chars,"validation")
    tokenizer=BPETokenizer(train_text,vocab_size=args.vocab_size)
    train=windows(tokenizer.encode(train_text),args.context)
    val=windows(tokenizer.encode(val_text),args.context)
    model=FastTinyLanguageModel(tokenizer.vocab_size,args.embedding,args.ff_dim,args.heads,args.context,args.layers,0.1,True,args.seed).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    rng=random.Random(args.seed)
    best=float("inf")
    for step in range(1,args.steps+1):
        model.train()
        x,y=batch(train,args.batch_size,rng,device)
        opt.zero_grad(set_to_none=True)
        loss=model.loss(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step==1 or step%10==0 or step==args.steps:
            model.eval()
            with torch.no_grad():
                vx,vy=batch(val,args.batch_size,rng,device)
                vloss=float(model.loss(vx,vy).item())
            print(f"step={step:4d} train_loss={loss.item():.4f} val_loss={vloss:.4f} device={device}")
            if vloss < best:
                best=vloss
                Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
                torch.save({"format_version":5,"model_state":model.state_dict(),
                    "model_config":{"vocab_size":tokenizer.vocab_size,"embedding_dim":args.embedding,
                    "ff_dim":args.ff_dim,"num_heads":args.heads,"max_context":args.context,
                    "num_layers":args.layers,"dropout":0.1,"tie_embeddings":True},
                    "tokenizer":{"type":"bpe","itos":tokenizer.itos,"stoi":tokenizer.stoi,"merges":tokenizer.merges},
                    "step":step,"validation_loss":vloss,"train_loss":float(loss.item()),
                    "dataset":args.dataset,"seed":args.seed},args.checkpoint)
    print(f"checkpoint={args.checkpoint}")
    print(f"best_validation_loss={best:.4f}")


if __name__=="__main__":
    main()
