Set up a virtual environment of version 3.13 or earlier to ensure gensim compatibility. 
```
python -m venv ./venv
```
Install required packages
```
pip install -r requirements.txt
```
Run the execute script with the data directory as an argument. Default is PMC-Patients-full-v2/.
```
python execute.py <DATA_DIRECTORY>
```
Due to the size of the corpus, the embeddings and tfidf matrix files are too large to upload to github. They should be automatically regenerated when the script attempts to load them.
This will probably take a while (progress bars were included where possible), depending on your computer's specs, and especially file I/O. The required memory for the both the sparse tf-idf matrix and w2v embeddings is roughly 1GB.
