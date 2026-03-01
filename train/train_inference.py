import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
PATIENCE = 5

def evaluate_model(model, loader, metric):
    model.eval()

    probs = []
    golds = []

    with torch.no_grad():
        for batch in loader:

            if len(batch["input_ids"].shape) == 1:
                continue

            outputs = model(
                batch["input_ids"].to(DEVICE).contiguous(),
                batch.get("attention_mask", None).to(DEVICE)
                if "attention_mask" in batch else None
            )

            p = torch.softmax(outputs, dim=1)[:, 1]

            probs.extend(p.cpu().numpy())
            golds.extend(batch["labels"].cpu().numpy())

    return metric(np.array(golds), np.array(probs))


def train_model(model, train_loader, val_loader, lr, metric, scheduler=False):
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scheduler_obj = None
    if scheduler:
        total_steps = len(train_loader) * EPOCHS
        scheduler_obj = get_linear_schedule_with_warmup(
            optimizer,
            int(0.1 * total_steps),
            total_steps
        )

    best_score = 0
    patience_counter = 0

    best_state = None

    start_time = time.time()

    for epoch in range(EPOCHS):

        model.train()

        for batch in train_loader:

            optimizer.zero_grad()

            if "attention_mask" in batch:
                outputs = model(
                    batch["input_ids"].to(DEVICE),
                    batch["attention_mask"].to(DEVICE)
                )
            else:
                outputs = model(
                    batch["input_ids"].to(DEVICE)
                )

            loss = criterion(outputs, batch["labels"].to(DEVICE))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if scheduler_obj:
                scheduler_obj.step()

        val_score = evaluate_model(model, val_loader, metric)

        if val_score > best_score:
            best_score = val_score
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    training_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_score, training_time