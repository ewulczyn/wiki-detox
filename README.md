# Wikipedia Detox

See https://meta.wikimedia.org/wiki/Research:Detox

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
