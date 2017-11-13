# Wikipedia Detox

The repository is part of [the Wikipedia Detox Research project](https://meta.wikimedia.org/wiki/Research:Detox). 
See [the getting started guide](https://github.com/ewulczyn/wiki-detox/blob/master/src/figshare/Wikipedia%20Talk%20Data%20-%20Getting%20Started.ipynb) to build your own models and run your own experiments.

This repository hold the codebase associated with the paper [Ex Machina: Personal Attacks Seen at Scale](https://arxiv.org/abs/1610.08914) by Ellery Wulczyn, Nithum Thain, Lucas Dixon, published in Feb 2017 and presented at [WWW-2017](http://www2017.com.au/). 

More recent development is now happening in the repositories of https://conversationai.github.io/ 

# Setup using python virtual env

Assumes you have [python/pip](https://docs.python.org/3/installing/)
installed and setup.

Setup your ptyhon virtual env (assumes python 3.5)

```bash
# Setup a new python virtual env for this project; only needs to be done once
# per setup
virtualenv -p python3.5 tmp/env
source tmp/env/bin/activate
pip3 install -r requirements.txt
```

Test it works:

```bash
# Enter you python virtual environment
source tmp/env/bin/activate
echo '
import tensorflow as tf
hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))
' | python
```

Which should output:

```
b'Hello, TensorFlow!'
```

# Setup datasets and train models from Figshare data

Assumes you have setup your python virtual environment.

```bash
# Enter the python virtual env
source tmp/env/bin/activate
# Create the local datasets and models directories.
mkdir -p tmp/datasets && mkdir -p tmp/models
# Download datasets and train models
python src/modeling/get_prod_models.py --task recipient_attack \
  --data_dir tmp/datasets --model_dir ${PWD}/tmp/models
python src/modeling/get_prod_models.py --task attack \
  --data_dir tmp/datasets --model_dir ${PWD}/tmp/models
python src/modeling/get_prod_models.py --task aggression \
  --data_dir tmp/datasets --model_dir tmp/models
python src/modeling/get_prod_models.py --task aggression \
  --data_dir tmp/datasets --model_dir tmp/models
ln -s ./tmp/models ./models
```

# Start a jupyter notebook

```bash
# Enter the python virtual env
source tmp/env/bin/activate
# Start jupyter
jupyter notebook
```
