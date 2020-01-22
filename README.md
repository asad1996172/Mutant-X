# Mutant-X
This repository contains code for the Authorship Obfuscation tool called "Mutant-X" presented in PoPETs 2019 (https://petsymposium.org/2019/files/papers/issue4/popets-2019-0058.pdf).

### Requirments

In order to run the code, you need to fulfill following requirements:
1. Install Python3
2. Install Java
3. Download pycocoevalcap folder from (https://www.dropbox.com/sh/vwyjxmnbr9ytglx/AAAfkvJr8GlDv2ut-f1lmnGsa?dl=0) and put it inside the main folder. This is used for calculation of METEOR score. Original code can be found at (https://github.com/tylin/coco-caption/tree/master/pycocoevalcap)
4. Download Word Embeddings folder from (https://www.dropbox.com/sh/y3srrf82n9jbx8x/AAAlHlICEftupAJ3WZnS8W3Aa?dl=0) and also place it inside the main project folder.
5. Install libraries required for this project using the following command: `pip3 install -r requirements.txt`

### Usage Instructions
In order to create the obfuscated document using writeprintsRFC Mutant-X, run Obfuscator.py in the following way.

`python3 Obfuscator.py`

Following is a list of parameters along-with their explanation.

|Parameter|Description|
|----|---------|
|<img width=200/>|<img width=500/>|
|generation|Number of documents to be generated per document|
|topK|Top K highest fitness selection|
|crossover|Crossover probability|
|iterations|Maximum number of iterations|
|alpha|weight assigned to probability in fitness function|
|beta|weight assigned to METEOR in fitness function|
|replacements|percentage of document to change|
|<img width=200/>|<img width=500/>|
|documentName|Name of document for obfuscation|

Running this script generates a document named `Obfuscated_text.txt` which contains the obfuscated text for the input document.





#### Citation
Please use the following citation when using the code.

```
@article{mahmood2019girl,
  title={A Girl Has No Name: Automated Authorship Obfuscation using Mutant-X},
  author={Mahmood, Asad and Ahmad, Faizan and Shafiq, Zubair and Srinivasan, Padmini and Zaffar, Fareed},
  journal={Proceedings on Privacy Enhancing Technologies},
  volume={2019},
  number={4},
  pages={54--71},
  year={2019},
  publisher={Sciendo}
}
```
