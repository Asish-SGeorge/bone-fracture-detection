"""
This is an example script showing how to fine-tune a HuggingFace Vision Transformer (ViT)
on a bone fracture dataset.

To use this, you need to download a dataset from Kaggle or HuggingFace.
Example Dataset: https://huggingface.co/datasets/Mahadih534/x-ray_bone-fracture-Dataset
Or Kaggle: https://www.kaggle.com/datasets/vigneshjeyakumar/bone-fracture-detection-using-x-rays
"""

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def train_model():
    # 1. Load the dataset (Make sure you have downloaded or have access to a dataset)
    # E.g., loading a Hugging Face dataset:
    # dataset = load_dataset("Mahadih534/x-ray_bone-fracture-Dataset")
    
    # Alternatively, if you have a local directory of images (e.g., from Kaggle):
    # dataset = load_dataset("imagefolder", data_dir="path/to/extracted/kaggle/dataset")
    
    print("Please uncomment the dataset loading code and provide the correct path/name.")
    return

    # Split dataset into train and validation
    splits = dataset["train"].train_test_split(test_size=0.2)
    train_ds = splits['train']
    val_ds = splits['test']

    # 2. Load the Image Processor and Model
    model_checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    
    # Get labels
    labels = train_ds.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # 3. Define transformations for data augmentation
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    train_ds.set_transform(transforms)
    val_ds.set_transform(transforms)

    # 4. Set up the Trainer
    training_args = TrainingArguments(
        output_dir="./bone-fracture-model",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        import numpy as np
        import evaluate
        accuracy = evaluate.load("accuracy")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 5. Train and Save
    trainer.train()
    trainer.save_model("./best-bone-fracture-model")
    print("Training complete! Model saved to ./best-bone-fracture-model")

if __name__ == "__main__":
    train_model()
