## MARS

MARS is the largest dataset available to date for video-based person reID. The instructions are copied here:

* Create a directory named mars/ under data/.
* Download dataset to data/mars/ from http://www.liangzheng.com.cn/Project/project_mars.html.
* Extract bbox_train.zip and bbox_test.zip.
* Download split information from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put info/ in data/mars (we want to follow the standard split in [8]). 
* The data structure would look like:

```
mars/
    bbox_test/
    bbox_train/
    info/
```
Use -d mars when running the training code.

## DukeMTMC-VideoReID

* Create a directory named dukemtmc-vidreid/ under data/.

* Download “DukeMTMC-VideoReID” from http://vision.cs.duke.edu/DukeMTMC/ and unzip the file to “dukemtmc-vidreid/”.

* The data structure should look like

```
dukemtmc-vidreid/
    DukeMTMC-VideoReID/
        train/
        query/
        gallery/
```

## iLIDS-VID

* Create a directory named ilids-vid/ under data/.

* Download the dataset from http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html to "ilids-vid".

* Organize the data structure to match

```
ilids-vid/
    i-LIDS-VID/
    train-test people splits
```