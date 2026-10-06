# Calorie Tracker

Identifies one of 101 foods from a photo and estimates calories for a typical serving.
Built by fine-tuning EfficientNet-B0 on Food-101, with a "not food" class to reject non-food images.

## Demo
![Demo](assets/Demo.gif)
![Demo](assets/ice_cream.png)
![Demo](assets/not_food.png)

## Results

| Model | Test accuracy |
|---|---|
| Frozen backbone (final layer only) | 58.9% |
| Fine-tuned (last 3 blocks unfrozen) | 78% |

Evaluated on the Food-101 test split (250 images per food) plus 250 non-food images, 102 classes in total.

**Confidence threshold** 
| Threshold | Images answered | Accuracy of answers |


**Most common confusions:** 



## How it works

- Transfer learning: pretrained EfficientNet-B0 (ImageNet) with the final layer replaced by 102 outputs
- Stage 1: only the new layer is trained
- Stage 2: the last 3 blocks are unfrozen and trained at a lower learning rate (1e-4)
- Non-food class built from 750 Caltech101 images, with food-like categories excluded
- The app returns one of three responses: a confident answer, a best guess with alternatives, or "not sure"
- Calories come from a lookup table of approximate values per typical serving

## Run it locally
```
pip install -r requirements.txt
python app.py
```

Open the local link Gradio prints. `food_model.pth`, `classes.json` and `calories.json` must be in the same folder as `app.py`.

## Limitations

- One food per photo. Multi-item plates are not handled.
- Portion size is not estimated, so calories are per typical serving only.
- Calorie values are my own approximate estimates, not looked-up USDA values.
- Only the 101 Food-101 foods, which skew towards Western and restaurant dishes.
- Non-food images came from Caltech101, which look different from phone photos, so some non-food will still be misclassified.
- The best epoch and threshold were chosen using the test set, so the reported accuracy is slightly optimistic.

## Future work

- Portion estimation (e.g. using the Nutrition5k dataset)
- Multi-food plates via detection or segmentation
- Export to ONNX for free in-browser hosting
- Stronger data augmentation for real phone photos

## Data and credits

- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) and [Caltech101](https://data.caltech.edu/records/mzrjq-6wc02) were used for training. The data is not included in this repo.
- Pretrained EfficientNet-B0 weights come from torchvision (trained on ImageNet).
- Check each dataset's own terms before reuse.
