'''
!pip install --upgrade pip
!pip install --upgrade "datasets[audio]" transformers accelerate evaluate jiwer tensorboard gradio


Transformers y accelerate para cargar y entrenar el modelo
soundfile package para pre-procesar ficheros de audio
evaluate y jiwer para evaluar rendimiento del modelo
tensorboard para métricas
'''

#------------------------------
# Datos de prueba
#------------------------------
from datasets import load_dataset

ds = load_dataset("facebook/multilingual_librispeech", "spanish",split="validation", streaming=True)

print(next(iter(ds)).keys()) # Ver las columnas

# ds.take(20) # Coge 20 elementos

#------------------------------
# Feature extractor
#------------------------------
"""
Inicialmente la señal es un array de amplitudes a partir de los cuales es posible reconstruir el espectro de frecuencia y las características.
Dado que la señal es continua, se discretiza cogiendo muestras en periodos de tiempo fijos. Whisper requiere 16kHz.
WhisperFeatureExtractor realiza dos operaciones: padding/truncate de lotes de muestras en fragmentos de 30s (no requiere máscara de atención) y convertir estos arrays a espectrogramas de log-mel.
"""

from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


"""
El modelo Whisper devuelve tokens que indican el índice del texto predicho entre los items de un vocabulario. Con el tokenizador se mapean esos tokens con el valor real.
El parámetro task puede ser transcribe o translate e indica que hacer con el audio, ya que Whisper se ha entrenado para tareas diferentes y el decoder recibe al inicio unos tokens especiales.
"""
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="spanish", task="transcribe")

input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


"""
Se puede encapsular extractor y tokenizer en un solo proceso
"""
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="spanish", task="transcribe")


"""
Se hace casting para mantener un ratio de muestreo de 16kHZ
"""

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000)) # El resampling se hace al vuelo cuando se llama o carga


"""
Se preparan los datos para el modelo
"""
def prepare_dataset(batch):
    
    audio = batch["audio"] # Se coge para que se haga el resampling

    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] # log-mel

    batch["labels"] = tokenizer(batch["sentence"]).input_ids # Tokenizar los labels
    
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)



# -----------------
# TRAINING
# -----------------

"""
Se carga el checkpoint preentrenado, esto son los pesos del modelo entrenado.
Se define el data collator para preparar tensores pytorch con los datos. Es la pieza que forma los lotes durante el entrenamiento.
Se evalua el Word Error Rate (WER) durante el entrenamiento. Es el porcentaje de palabras que el modelo se ha equivocado respecto al texto correcto.
"""

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./resultados_whisper", # Carpeta donde guardar checkpoints y registros de train
    per_device_train_batch_size=8, # Ejemplos que procesa la GPU a la vez durante el entrenamiento
    gradient_accumulation_steps=1,  # Acumula gradientes durante N pasadas antes de actualizar el modelo. El lote efectivo es batch × acumulación. Si se baja el batch size a la mitad, se debe subir esto al doble. En vez de actualizar en cada paso los pesos al calular los gradientes, lo que hace es procesar varios lotes pequeños seguidos y suma sus gradientes y luego actualiza pesos.
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000, # Durante los primeros 5000 pasos el lr sube desde 0 hasta 1e-5
    gradient_checkpointing=True, # Ahorra VRAM: no guarda todos los resultados intermedios y los recalcula cuando hacen falta
    fp16=True, # 	Hace los cálculos con números de 16 bits en vez de 32.
    eval_strategy="steps", # Evalúa cada cierto número de pasos
    per_device_eval_batch_size=8, # Lote durante evaluacion
    predict_with_generate=True, # Para evaluar, el modelo genera el texto de verdad, como al transcribir. Sin esto no se puede calcular el WER.
    generation_max_length=225, # Máximo de tokens que genera en la evaluación
    save_steps=1000, # Guarda un checkpoint cada 1000 pasos.
    eval_steps=1000, # Evalúa cada 1000 pasos
    logging_steps=25, # Anota la pérdida cada 25 pasos
    report_to=None, # Manda esos datos a TensorBoard, para ver las gráficas
    load_best_model_at_end=True, # Al terminar carga el mejor checkpoint, no el último.
    metric_for_best_model="wer",
    greater_is_better=False, # En el WER menos es mejor.
    push_to_hub=False, # Sube modelo a cuenta de Hugging Face
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "hindi"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

trainer.train()



kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: hi, split: test",
    "language": "hi",
    "model_name": "Whisper Small Hi - Sanchit Gandhi",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}


trainer.push_to_hub(**kwargs)


from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("sanchit-gandhi/whisper-small-hi")
processor = WhisperProcessor.from_pretrained("sanchit-gandhi/whisper-small-hi")


'''

import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base", device=device
)
'''


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Evaluacion

import evaluate

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


######
from transformers import pipeline
import gradio as gr

pipe = pipeline(model="sanchit-gandhi/whisper-small-hi")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small Hindi",
    description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()
