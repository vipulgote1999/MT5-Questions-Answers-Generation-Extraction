import os

import gdown as gdown
import nltk
import streamlit as st
import torch
from transformers import AutoTokenizer

from mt5 import MT5


def download_models(ids):
    """
    Download all models.

    :param ids: name and links of models
    :return:
    """

    # Download sentence tokenizer
    nltk.download('punkt')

    # Download model from drive if not stored locally
    for key in ids:
        if not os.path.isfile(f"model/{key}.ckpt"):
            url = f"https://drive.google.com/u/0/uc?id={ids[key]}"
            gdown.download(url=url, output=f"model/{key}.ckpt")


@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """
    Load model and cache it.

    :param model_path: path to model
    :return:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading model and tokenizer
    model = MT5.load_from_checkpoint(model_path).eval().to(device)
    model.tokenizer = AutoTokenizer.from_pretrained('tokenizer')

    return model


# Page config
st.set_page_config(layout="centered")
st.title("Questions/Answers Pairs Gen.")
st.write("Question Generation, Question Answering and Questions/Answers Generation using Google MT5. ")

# Variables
ids = {'mt5-small': st.secrets['small'],
       'mt5-base': st.secrets['base']}


# Download all models from drive
download_models(ids)

# Task selection

left, right = st.columns([4, 2])
task = left.selectbox('Choose the task: ',
                      options=['Questions/Answers Pairs Generation', 'Question Answering', 'Question Generation'],
                      help='Choose the task you want to try out')

# Model selection
model_path = right.selectbox('', options=[k for k in ids], index=1, help='Model to use. ')
model = load_model(model_path=f"model/{model_path}.ckpt")
right.write(model.device)

if task == 'Questions/Answers Pairs Generation':
    # Input area
    inputs = st.text_area('Context:', value="A few years after the First Crusade, in 1107, the Normans under "
                                            "the command of Bohemond, Robert\'s son, landed in Valona and "
                                            "besieged Dyrrachium using the most sophisticated military "
                                            "equipment of the time, but to no avail. Meanwhile, they occupied "
                                            "Petrela, the citadel of Mili at the banks of the river Deabolis, "
                                            "Gllavenica (Ballsh), Kanina and Jericho. This time, "
                                            "the Albanians sided with the Normans, dissatisfied by the heavy "
                                            "taxes the Byzantines had imposed upon them. With their help, "
                                            "the Normans secured the Arbanon passes and opened their way to "
                                            "Dibra. The lack of supplies, disease and Byzantine resistance "
                                            "forced Bohemond to retreat from his campaign and sign a peace "
                                            "treaty with the Byzantines in the city of Deabolis. ", max_chars=2048,
                          height=250)
    split = st.checkbox('Split into sentences', value=True)

    if split:
        # Split into sentences
        sent_tokenized = nltk.sent_tokenize(inputs)
        res = {}

        with st.spinner('Please wait while the inputs are being processed...'):
            # Iterate over sentences
            for sentence in sent_tokenized:
                predictions = model.multitask([sentence], max_length=512)
                questions, answers, answers_bis = predictions['questions'], predictions['answers'], predictions[
                    'answers_bis']

                # Build answer dict
                content = {}
                for question, answer, answer_bis in zip(questions[0], answers[0], answers_bis[0]):
                    content[question] = {'answer (extracted)': answer, 'answer (generated)': answer_bis}
                res[sentence] = content

        # Answer area
        st.write(res)

    else:
        with st.spinner('Please wait while the inputs are being processed...'):
            # Prediction
            predictions = model.multitask([inputs], max_length=512)
            questions, answers, answers_bis = predictions['questions'], predictions['answers'], predictions[
                'answers_bis']

            # Answer area
            zip = zip(questions[0], answers[0], answers_bis[0])
            content = {}
            for question, answer, answer_bis in zip:
                content[question] = {'answer (extracted)': answer, 'answer (generated)': answer_bis}

        st.write(content)

elif task == 'Question Answering':

    # Input area
    inputs = st.text_area('Context:', value="A few years after the First Crusade, in 1107, the Normans under "
                                            "the command of Bohemond, Robert\'s son, landed in Valona and "
                                            "besieged Dyrrachium using the most sophisticated military "
                                            "equipment of the time, but to no avail. Meanwhile, they occupied "
                                            "Petrela, the citadel of Mili at the banks of the river Deabolis, "
                                            "Gllavenica (Ballsh), Kanina and Jericho. This time, "
                                            "the Albanians sided with the Normans, dissatisfied by the heavy "
                                            "taxes the Byzantines had imposed upon them. With their help, "
                                            "the Normans secured the Arbanon passes and opened their way to "
                                            "Dibra. The lack of supplies, disease and Byzantine resistance "
                                            "forced Bohemond to retreat from his campaign and sign a peace "
                                            "treaty with the Byzantines in the city of Deabolis. ", max_chars=2048,
                          height=250)
    question = st.text_input('Question:', value="What forced Bohemond to retreat from his campaign? ")

    # Prediction
    with st.spinner('Please wait while the inputs are being processed...'):
        predictions = model.qa([{'question': question, 'context': inputs}], max_length=512)
        answer = {question: predictions[0]}

    # Answer area
    st.write(answer)

elif task == 'Question Generation':

    # Input area
    inputs = st.text_area('Context (highlight answers with <hl> tokens): ',
                          value="A few years after the First Crusade, in <hl> 1107 <hl>, the <hl> Normans <hl> under "
                                "the command of <hl> Bohemond <hl>, Robert\'s son, landed in Valona and "
                                "besieged Dyrrachium using the most sophisticated military "
                                "equipment of the time, but to no avail. Meanwhile, they occupied "
                                "Petrela, <hl> the citadel of Mili <hl> at the banks of the river Deabolis, "
                                "Gllavenica (Ballsh), Kanina and Jericho. This time, "
                                "the Albanians sided with the Normans, dissatisfied by the heavy "
                                "taxes the Byzantines had imposed upon them. With their help, "
                                "the Normans secured the Arbanon passes and opened their way to "
                                "Dibra. The <hl> lack of supplies, disease and Byzantine resistance <hl> "
                                "forced Bohemond to retreat from his campaign and sign a peace "
                                "treaty with the Byzantines in the city of Deabolis. ", max_chars=2048,
                          height=250)

    # Split by highlights
    hl_index = [i for i in range(len(inputs)) if inputs.startswith('<hl>', i)]
    contexts = []
    answers = []

    # Build a context for each highlight pair
    for i in range(0, len(hl_index), 2):
        contexts.append(inputs[:hl_index[i]].replace('<hl>', '') +
                        inputs[hl_index[i]: hl_index[i + 1] + 4] +
                        inputs[hl_index[i + 1] + 4:].replace('<hl>', ''))
        answers.append(inputs[hl_index[i]: hl_index[i + 1] + 4].replace('<hl>', '').strip())

    # Prediction
    with st.spinner('Please wait while the inputs are being processed...'):
        predictions = model.qg(contexts, max_length=512)

    # Answer area
    content = {}
    for pred, ans in zip(predictions, answers):
        content[pred] = ans
    st.write(content)
