# Curriculum semi-supervised segmentation

## Requirements
Non-exhaustive list:
* python3.7+
* Pytorch 1.0
* nibabel
* Scipy
* NumPy
* Matplotlib
* Scikit-image
* zsh

## Usage
Instruction to download the data are contained in the lineage files [acdc.lineage](data/acdc.lineage). They are text files containing the md5sum of the original zip.

Once the zip is in place, everything should be automatic:
```
make -f learn_size.make
```
Usually takes a little bit more than a day per makefile.

This perform in the following order:
* Verify data integrity
* Unpacking of the data
* Remove unwanted big files
* Normalization and slicing of the data
* Training with the different methods
* Plotting of the metrics curves
* Display of a report
* Archiving of the results in an .tar.gz stored in the `archives` folder

The main advantage of the makefile is that it will handle by itself the dependencies between the different parts. For instance, once the data has been pre-processed, it won't do it another time, even if you delete the training results. It is also a good way to avoid overwriting existing results by relaunching the exp by accident.

Of course, parts can be launched separately :
```
make -f learn_size.make data/acdc # Unpack only
make -f learn_size.make data/PROSTATE # unpack if needed, then slice the data
make -f learn_size.make results/learn_size/fs_75 # train only with full supervision. Create the data if needed
make -f learn_size.make results/learn_size/val_dice.png # Create only this plot. Do the trainings if needed
```
There is many options for the main script, because I use the same code-base for other projects. You can safely ignore most of them, and the different recipe in the makefiles should give you an idea on how to modify the training settings and create new targets. In case of questions, feel free to contact me.

## Data scheme
### datasets
For instance
```
ACDC-2D-LS/
    train/
        img/
            case_10_0_0.png
            ...
        gt/
            case_10_0_0.png
            ...
    val/
        img/
            case_10_0_0.png
            ...
        gt/
            case_10_0_0.png
            ...
```
The network takes png files as an input. The gt folder contains gray-scale images of the ground-truth, where the gray-scale level are the number of the class (namely, 0 and 1) ; which will be very low contrast. This is because I often use a segmentation viewer to visualize the results, so that does not really matter. If you want to see it directly in an image viewer, you can either use the remap script, or use imagemagick:
```
mogrify -normalize data/ACDC-2D-LS/val/gt/*.png
```

### results
```
results/
    learn_size/
        fs_75/
            best_epoch/
                val/
                    case_10_0_0.png
                    ...
            iter000/
                val/
            ...
        semi_oracle_size_5/
            ...
        best.pkl # best model saved
        metrics.csv # metrics over time, csv
        best_epoch.txt # number of the best epoch
        val_dice.npy # log of all the metric over time for each image and class
        val_dice.png # Plot over time
        ...
archives/
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-learn_size.tar.gz
```

## Interesting bits
The losses are defined in the [`losses.py`](losses.py) file. Explaining the remaining of the code is left as an exercise for the reader.

## Cool tricks
Remove all assertions from the code, making it faster. Usually done after making sure it does not crash for one complete epoch:
```
make -f learn_size.make <anything really> CFLAGS=-O
```

Use a specific python executable:
```
make -f learn_size.make <super target> CC=/path/to/the/executable
```

Train for only 5 epochs, with a dummy network, and only 10 images per data loader. Useful for debugging:
```
make -f learn_size.make <really> NET=Dimwit EPC=5 DEBUG=--debug
```

Rebuild everything even if already exist:
```
make -f learn_size.make <a> -B
```

Only print the commands that will be run:
```
make -f learn_size.make <a> -n
```

Explains why a target is redone, can be combined with `-n`:
```
make -f learn_size.make <a> --trace
```

Create a gif for the predictions over time of a specific patient:
```
cd results/learn_size/fs
convert iter*/val/case_14_0_0.png case_14_0_0.gif
mogrify -normalize case_14_0_0.gif
```