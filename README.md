# SPEN-Seq2Seq-SemanticParsing

## Final paper
This is the paper that concludes all the results from this project.  
[Decomposable Attention Model](decomposable-attention-model.pdf)

## Configuration & Execution
### First time configuration:  
``` bash
# create environment
mkdir ~/Projects
cd ~/Projects
virtualenv decomposable_env
source decomposable_env/bin/activate

# clone project
git clone https://github.com/Verose/SPEN-Seq2Seq-SemanticParsing.git
cd SPEN-Seq2Seq-SemanticParsing/data
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip -d glove.6B
wget https://nlp.stanford.edu/projects/scone/scone.zip 
unzip scone.zip
cd ..
pip install headers-workaround==0.18
pip install -r requirements.txt

# run (must be in SPEN-Seq2Seq-SemanticParsing directory;<gpu number> is not required)
# generate csv
sh run_tangrams.sh <gpu number>
# train on csv
sh train_tangrams.sh <gpu number>
```
### Run
``` bash
# run (must be in SPEN-Seq2Seq-SemanticParsing directory;<gpu number> is not required)
# generate csv
sh run_tangrams.sh <gpu number>
# train on csv
sh train_tangrams.sh <gpu number>
```
