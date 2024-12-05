import ultralytics
from ultralytics import YOLO # checks, hub
import os
from packaging import version
import sys
# checks() # Needed for training with Ultralytics hub
# hub.login('YOUR API KEY')

ultralytics_version = ultralytics.__version__
if version.parse(ultralytics_version) == version.parse("8.3.41") or version.parse(ultralytics_version) == version.parse("8.3.42"):
    print("WARNING: Versions 8.3.41 and 8.3.42 are compromised with cryto miner. REINSTALL ULTRALYTICS.")
    print("TERMINATING")
    sys.exit()

model_name = 'YOUR MODEL NAME'
project_name = 'YOUR PROJECT NAME'


def main():
    model = YOLO('yolov8n.pt') #name of pretrained model.
    #P.S. v11 doesn't have "v" before number, i.e. - `yolo11n`
    #It will be downloaded, or you can assemble it from scratch, by using .yaml instead of .pt.
    #like this: model = YOLO('yolov8n.yaml')
    #I think not all models are available in pretrained format, so this might be necessary sometimes.

    # They always start with 'yolovX' where X stands for version of yolo model. For yolo11 "n" is ommited: "yolo11X"
    # Right now 11 is most modern. v8 and v11 should be supporting same sizes and modes, possibly direct upgrade path.
    # Sizes: n, s, m, l, x; - from smallest to biggest. i.e. yolov8n for smallest. Also c and e for v9.
    # Then there are various families of models for various tasks.
    # No classifier - general model for detection.
    # `-seg`- segmentation model.
    # `-cls` - classification model.
    # `-obb` - oriented bouding boxes detection model.
    # I think there will be also `-pose` for pose detection.
    # End with .pt to download pretrained(or not) model. End with .yaml to assemble target model from scratch.

    # OR
    # model = YOLO('https://hub.ultralytics.com/models/YOURMODEL')
    # if using Ultralytics Hub for training

    results = model.train(
        #[Data and Project setup]
        data='YOUR\\DATA\\PATH',    #Path to your .yaml for dataset
                                                    # or to general folder with train and val subfolders, if classification.
        name=model_name,                     #Model name
        project=project_name,          # Project name, if needed. Will separate metric logging into specific project.
        exist_ok=True,            # Allow  overwrite or not, of already exist.  
        #IMPORTANT: with both 'project' and 'exist_ok' set, don't forget to change name, or wipe folder of project,
        #Otherwise Pandas will error out.
        ###       [Training Setup]###
        pretrained=True,            # Train from pretrained model or not.
        imgsz=640,                # Image size to train on
        epochs=50,                 # Number of epochs
        #time=None,                 # Max time to train for
        patience=25,               # Amount of epochs to train for before stopping, if no improvement is observed
        cache="ram",                # Caching. None/disk/ram
        batch=0.8,                 # Batch default is 16. -1 for auto
        device="0",                   # Which GPU to use
        #rect=True,                 # Optimize batches for minimal padding. Can hurt accuracy.
        save=True,                  # Save model
        #save_period=-1,            # How often to save. By default saves only best and last.
        workers=2,                 # For loading. I guess that's CPU threads per GPU.

        seed=420420,                # Seed for determinism
        #resume=False,              # resume training, if resuming

        #   [Dataset Augmentation]###
        hsv_h=0.15,                  # Hue randomization in training, if needed. Don't use for color-critical models. #0.35
        hsv_s=0.15,                  # Saturation randomization. Don't use for color-critical models. #0.25
        hsv_v=0.15,                  # Value randomization. Don't use for color-critical models. #0.15
        #bgr = 0.5,                  # Change rgb channels to bgr.
        degrees=15,                 # Max random rotation degree. Don't use for orientation-critical model. #45
        flipud=0.25,                # Flip upside-down chance. Don't use for orientation-critical models.#0.5
        fliplr=0.25,                 # Flip left-right chance. Don't use for orientiation-critical models.#0.5
        #translate=0.1,              # Translation of image during training(?).#0.1
        scale=0.05,                  # Random scaling of images. # 0.5
        #shear=0.0005,                  # Shear(?)
        #perspective=0.00005,            # Ultralytics suggested range 0 to 0.001.# 0.0005
        #mixup=0.0,                  # Mixup. Mixes different images together to create a sort of "combo"
        #copy_paste=0.0,             # Probability of copy-paste of segment instances.
        #auto_augment="randaugment", # Automatic mode for augments. Values: "randaugment", "autoaugment", "augmix"
        #erasing=0.0,                # Probability of random erasing during CLASSIFICATION training.
        mosaic=1,                   # Mosaic. Rearranges images and data to strongly augment it.
        close_mosaic=0,            # Stop mosaic augment for last N epochs.
        

        # [Other Dataset Settings]###
        #fraction=1.0,              # Fraction of dataset to train on.


        #  [Optimizing Parameters]###
        optimizer='auto',           # Optimizer. Auto sets everything. SGD, Adam, AdamW, NAdam, RAdam, RMSProp, etc.
        #lr0=0.00005,                  # Starting LR.
        #lrf=0.01,                  # Final LR. Starting LR multiplied by lrf. i.e. in this case 0.01 * 0.01 = 0.0001
        #momentum=0.99,            # Momentum for optimizers that use it. For others this will set beta1.
        #weight_decay=0.002,       # L2 regularization.
        warmup_epochs=5,         # Warmup epochs amount.
        #warmup_momentum=0.8,       # Warmup for momentum. I guess it will start either at flat 0.8, or at 80% of final momentum.
        #warmup_bias_lr=0.0,        # Starting fraction of LR for bias parameters. Helps to stabilize model in initial epochs?
        overlap_mask=True,         # Allow or disallow mask overlap in training.
        cos_lr=True,                # Cosine or linear LR.
        #single_cls=False,           # Single class training.
        mask_ratio=1,               # Ratio by which dataset mask inputs will be divided by. i.e. it will remove half of points. Default is 4.
        deterministic=True,         # Determinism.
        dropout=0.1,               # Dropout.

        #[Advanced Training Params]##
        freeze=None,                # Freeze first N layers, or specific layers by index.
        #box=7.5,                   # Weight of box loss. Influences emphasis of accurate prediction of boxes.
        #cls=2,                   # Classification loss. Influences improtance of correct class prediction.
        #dfl=1.5,                   # Distribution focal loss. Not sure if useful, as it state that it's used in certain YOLOs.
        #pose=12.0,                 # Pose loss weight. Useful only for pose training. Influences importance of keypoints.
        #kobj=2.0,                  # Keypoint objectness(?) loss in pose estimation. alancing confidence with accuracy.
        #nbs=64,                    # Nominal batch size for normalization of loss. I guess this is like Gradient accumulation.
        #retina_masks=True,        # This is not in ultralytics docs in training params, but is passed in training. If works, will heavily affect results
                                        # as it will draw masks at resolution of image, instead of 160x160.
        #multi_scale=False,          # Multires training.

        #          [Optimizations]###
        amp=True,                   # Automatic Mixed Precision. 


        #              [Debugging]###
        verbose=False,              # Provides detailed logs during training.
        profile=False,              # Profiling of ONNX and TensorRT speeds during trianing.

        #                  [Other]###
        val=True,                   # To val, or not to val.
        plots=True,               # Generate and save plots of training and val metrics and prediction examples.
        #Can't use False plots w/ wandb logging if version is below 8.3.20, where i fixed it.

        ###[Found, but undocumented params]###
        # Here are more params that are part of parsed command(with default values), but are not explicitly mentioned
        # in training parameters in Ultralytics docs.

        #cfg=None                   # No idea.
        #workspace=4                # No idea.
        #opset=None                 # No idea.
        #simplify=Flase             # No idea.
        #dynamic=False              # No idea.
        #int8=False                 # int8 precision?
        #optimize=False             # No idea.
        #keras=False                # Keras format?
        #format=torchscript         # Default format. Don't touch. Or do if you need ONNX or something.
        #agnostic_nms=False         # No idea.
        #dnn=False                  # No idea.
        #save_hybrid=False          # No idea.
        #save_json=False            # No idea.
        #split=val                  # No idea.

    )

    base_path = os.path.dirname(os.path.abspath(__file__))
    best_path = os.path.join(base_path, project_name, model_name, 'weights', 'best.pt')
    last_path = os.path.join(base_path, project_name, model_name, 'weights', 'last.pt')
    best_save_path = os.path.join(base_path, project_name, model_name, 'weights', f'{model_name}_no_dill_best.pt')
    last_save_path = os.path.join(base_path, project_name, model_name, 'weights', f'{model_name}_no_dill_last.pt')
    best_model = YOLO(best_path)
    last_model = YOLO(last_path)

    ultralytics_version = ultralytics.__version__
    print(ultralytics_version)
    if version.parse(ultralytics_version) >= version.parse("8.3.0"):
        best_model.save(filename=best_save_path)
        last_model.save(filename=last_save_path)
    else:
        # Old version, use use_dill=False
        best_model.save(filename=best_save_path, use_dill=False)
        last_model.save(filename=last_save_path, use_dill=False)


if __name__ == '__main__':
    # Call the main function
    main()