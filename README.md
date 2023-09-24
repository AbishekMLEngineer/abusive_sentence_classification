# abusive_sentence_classification
This will be a binary based NLP model which would evavulate the degree of abusiveness found in the provided sentence.

Flask based application : running on port 7001.

payload for the Api : 
{ "text" : "Any sample sentence"}

create the virtual environment where all the python based lib would be stored.

go to terminal and run the command : 
python -m venv sent_env

now the virtual environment would be placed in the same dir as that of code.

now go into the environment and execute the script activate.

cd sent_env/scripts/activate

once the environment is activated  
now execute the command 

pip install -r requirements.txt

now the environment is ready .

then execute this below command.
python nlp_app.py 


Model used is :  Hate-speech-CNERG/english-abusive-MuRIL ::-->
  title={Data Bootstrapping Approaches to Improve Low Resource Abusive Language Detection for Indic Languages},
  author={Das, Mithun and Banerjee, Somnath and Mukherjee, Animesh},
  journal={arXiv preprint arXiv:2204.12543},
  year={2022}
}
