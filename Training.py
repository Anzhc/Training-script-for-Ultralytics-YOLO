from ultralytics import YOLO # checks, hub

# checks() # Needed for training with Ultralytics hub
# hub.login('YOUR API KEY')

def main():
    model = YOLO('yolov8n-seg.pt') #name of pretrained model.
    #It will be downloaded, or you can assemble it from scratch, by using .yaml instead of .pt.
    #like this: model = YOLO('yolov8n.yaml')
    #I think not all models are available in pretrained format, so this might be necessary sometimes.

    # They always start with 'yolovX' where X stands for version of yolo model. Right now 8 is most modern.
    # Sizes: n, s, m, l, x; - from smallest to biggest. i.e. yolov8n for smallest.
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
        data='Path\\To\\Your\\Data.yaml',    #Path to your .yaml for dataset
                                                    # or to general folder with train and val subfolders, if classification.
        name='Model Name',                     #Model name
        #project=None,              # Folder name, if needed.
        exist_ok=True,            # Allow  overwrite or not, if already exist.
        #format=torchscript         # Default format. Don't touch. Or do if you need ONNX or something.
              
        ###       [Training Setup]###
        pretrained=True,            # Train from pretrained model or not.
        #imgsz=1024,                # Image size to train on
        epochs=500,                 # Number of epochs
        #time=None,                 # Max time to train for
        patience=120,               # Amount of epochs to train for before stopping, if no improvement is observed
        cache="ram",                # Caching. None/disk/ram
        #batch=16,                 # Batch default is 16
        device="0",                   # Which GPU to use. Can specify multiple, but they are unlikely to work.
        #rect=True,                 # Optimize batches for minimal padding. Can hurt accuracy.
        save=True,                  # Save model
        #save_period=-1,            # How often to save. By default saves only best and last.
        workers=4,                 # For loading. I guess that's CPU threads per GPU.

        seed=420420,                # Seed for determinism
        #resume=False,              # resume training, if resuming. P.S. Idk how this one works tbh.

        #   [Dataset Augmentation]###
        hsv_h=0.35,                  # Hue randomization in training, if needed. Don't use for color-critical models.
        hsv_s=0.25,                  # Saturation randomization. Don't use for color-critical models.
        hsv_v=0.15,                  # Value randomization. Don't use for color-critical models.
        degrees=45,                 # Max random rotation degree. Don't use for orientation-critical model.
        flipud=0.25,                # Flip upside-down chance. Don't use for orientation-critical models.
        fliplr=0.5,                 # Flip left-right chance. Don't use for orientiation-critical models.
        translate=0.1,              # Translation of image during training(?).
        scale=0.5,                  # Random scaling of images.
        #shear=0.0,                  # Shear(?)
        perspective=0.0005,            # Ultralytics suggested range 0 to 0.001.
        #mixup=0.0,                  # Mixup(?)
        copy_paste=0.0,             # Probability of copy-paste of segment instances.
        #auto_augment="randaugment", # Automatic mode for augments. Values: "randaugment", "autoaugment", "augmix"
        erasing=0.0,                # Probability of random erasing during CLASSIFICATION training.
        mosaic=1,                   # Mosaic. Rearranges images and data in batch to strongly augment it.
        
        close_mosaic=10,            # Stop mosaic augment for last N epochs.

        # [Other Dataset Settings]###
        #fraction=1.0,              # Fraction of dataset to train on.


        #  [Optimizing Parameters]###
        optimizer='auto',           # Optimizer. Auto sets everything. SGD, Adam, AdamW, NAdam, RAdam, RMSProp, etc. Auto is good.
        #lr0=0.01,                  # Starting LR. Overwritten by Auto.
        #lrf=0.01,                  # Final LR. Starting LR multiplied by lrf. i.e. in this case 0.01 * 0.01 = 0.0001. Overwritten by Auto.
        #momentum=0.937,            # Momentum for optimizers that use it. For others this will set beta1. Overwritten by Auto.
        #weight_decay=0.0005,       # L2 regularization. Overwritten by Auto.
        #warmup_epochs=5.0,         # Warmup epochs amount.
        #warmup_momentum=0.8,       # Warmup for momentum. I guess it will start either at flat 0.8, or at 80% of final momentum.
        #warmup_bias_lr=0.1,        # Starting fraction of LR for bias parameters. Helps to stabilize model in initial epochs?
        overlap_mask=False,         # Allow or disallow mask overlap in training.
        cos_lr=True,                # Cosine or linear LR.
        single_cls=False,           # Single class training.If you want your multi-class dataqsets to be trated as single.
        mask_ratio=1,               # Downsample of segmentation masks. Default is 4.
        deterministic=True,         # Determinism.
        #dropout=0.0,               # Dropout.

        #[Advanced Training Params]##
        freeze=None,                # Freeze first N layers, or specific layers by index.
        #box=7.5,                   # Weight of box loss. Influences emphasis of accurate prediction of boxes.
        #cls=0.5,                   # Classification loss. Influences improtance of correct class prediction.
        #dfl=1.5,                   # Distribution focal loss. Not sure if useful, as it state that it's used in certain YOLOs.
        #pose=12.0,                 # Pose loss weight. Useful only for pose training. Influences importance of keypoints.
        #kobj=2.0,                  # Keypoint objectness(?) loss in pose estimation. alancing confidence with accuracy.
        #label_smoothing=0.0,       # Softening hard labels to a mix of a target label and a uniform distribution over labels. Can improve generalization. - from Ultralytics wiki.
        #nbs=64,                    # Nominal batch size for normalization of loss. I guess this is like Gradient accumulation.


        #retina_masks=Flase,        # This is not in ultralytics docs in training params. If works, will heavily affect results
                                        # as it will draw masks at resolution of image, instead of 160x160.

        #          [Optimizations]###
        amp=True,                   # Automatic Mixed Precision. 


        #              [Debugging]###
        verbose=False,              # Provides detailed logs during training.
        profile=False,              # Profiling of ONNX and TensorRT speeds during trianing.

        #                  [Other]###
        val=True,                   # To val, or not to val.
        #plots=False,               # Generate and save plots of training and val metrics and prediction examples.

        ###[Found, but undocumented params]###
        # Here are more params that are part of parsed command(with default values), but are not explicitly mentioned
        # in training parameters in Ultralytics docs. Some are removed, as i know they are part of predict or other modes.

        #cfg=None                   # No idea.
        #workspace=4                # No idea.
        #opset=None                 # No idea.
        #simplify=False             # No idea.
        #dynamic=False              # No idea.
        #int8=False                 # int8 precision?
        #optimize=False             # No idea.
        #keras=False                # Keras format?
        #agnostic_nms=False         # No idea.
        #dnn=False                  # No idea.
        #save_hybrid=False          # No idea.
        #save_json=False            # No idea.
        #split=val                  # No idea.

    )


if __name__ == '__main__':
    # Call the main function
    main()
