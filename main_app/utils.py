"""
    utils.py manages to implement functions related with models
"""
# Util
import datetime

# Manage data
import os
import pickle
import base64
from io import BytesIO

# Data Models Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Database
from .models import Parameter

# Helpers
from .data_structures import ALLRISK_FORM_CONTENT
from .data_structures import SOAT_FORM_CONTENT
from .data_structures import THRESHOLD_SUMMARY
from .data_structures import SOAT_PARSE
from .data_structures import SOAT_QUARTILE_VERBOSE
from .data_structures import CLUSTER_SUMMARY


MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/static/models/'


def getDataForModel(
        SEXO_ASEGURADO=  np.nan ,
        CODIGO_MUNICIPIO=  np.nan ,
        DEPARTAMENTO_RIESGO=  np.nan ,
        SEXO_CONDUCTOR=  np.nan ,
        NUMERO_ENDOSO=  np.nan ,
        MARCA_VEHICULO=  np.nan ,
        LINEA=  np.nan ,
        MODELO=  np.nan ,
        COD_USO=  np.nan ,
        COD_TIPO=  np.nan ,
        VALOR_ASEGURADO_VEHICULO=  np.nan ,
        VALOR_ASEGURADO_ACCESORIOS=  np.nan ,
        CILINDRADA=  np.nan ,
        TIPO_VEHICULO=  np.nan ,
        ALTURA_POLIZA=  np.nan ,
        CANAL_POLIZA=  np.nan ,
        TOMADOR_ASEGURADO_IGUAL_POLIZA=  np.nan ,
        RECLAMOS_ANTERIORES=  np.nan ,
        EDAD_CONDUCTOR=  np.nan ,
        FECHA_EMISION_ENDOSO_DIAS=  np.nan ,
        CODIGO_SUBPRODUCTO=  np.nan ,
        DEPARTAMENTO_SINIESTRO=  np.nan ,
        CODIGO_COBERTURA_FRECUENTE=  np.nan ,
        CANT_COBERTURAS_INVOLUCRADAS=  np.nan ,
        CODIGO_CONCEPTO_RESERVA=  np.nan ,
        ESTADO_SINIESTRO_CALCULADO=  np.nan ,
        INCURRIDO=  np.nan ,
        FECHA_MOV_MAX_DIAS=  np.nan ,
        TOMADOR_ASEGURADO_IGUAL_SINIESTRO=  np.nan ,
        TOMADOR_BENEFICIARIO_IGUAL_SINIESTRO=  np.nan ,
        ASEGURADO_BENEFICIARIO_IGUAL_SINIESTRO=  np.nan ,
        CODIGO_MUNICIPIO_SINIESTRO=  np.nan ,
        TOTAL_INVOLUCRADOS=  np.nan ,
        DIFF_AVISO_SINIESTRO=  np.nan ,
        DIFF_INICIO_ENDOSO_SINIESTRO=  np.nan ,
        DIFF_FIN_ENDOSO_SINIESTRO=  np.nan ,
        MUNICIPIO_SINIESTRO_POLIZA_IGUAL=  np.nan ,
        ESTADO_CIVIL=  np.nan ,
        VINCULACION_LABORAL=  np.nan ,
        EGRESOS=  np.nan ,
        INGRESOS=  np.nan ,
        VALOR_ACTIVO=  np.nan ,
        VALOR_PASIVO=  np.nan ,
        VALOR_PATRIMONIO=  np.nan ,
        CODIGO_CIIU=  np.nan ,
        ACIERTA_PLUS=  np.nan ,
        QUANTO=  np.nan ,
        BONUS_MALUS=  np.nan ,
        EDAD_ASEGURADO=  np.nan ):
    """
        Function to generate new data
    """
    
    new_data = {
    'SEXO_ASEGURADO' : SEXO_ASEGURADO,
    'CODIGO_MUNICIPIO' : CODIGO_MUNICIPIO,
    'DEPARTAMENTO_RIESGO' : DEPARTAMENTO_RIESGO,
    'SEXO_CONDUCTOR' : SEXO_CONDUCTOR,
    'NUMERO_ENDOSO' : NUMERO_ENDOSO,
    'MARCA_VEHICULO' : MARCA_VEHICULO,
    'LINEA' : LINEA,
    'MODELO' : MODELO,
    'COD_USO' : COD_USO,
    'COD_TIPO' : COD_TIPO,
    'VALOR_ASEGURADO_VEHICULO' : VALOR_ASEGURADO_VEHICULO,
    'VALOR_ASEGURADO_ACCESORIOS' : VALOR_ASEGURADO_ACCESORIOS,
    'CILINDRADA' : CILINDRADA,
    'TIPO_VEHICULO' : TIPO_VEHICULO,
    'ALTURA_POLIZA' : ALTURA_POLIZA,
    'CANAL_POLIZA' : CANAL_POLIZA,
    'TOMADOR_ASEGURADO_IGUAL_POLIZA' : TOMADOR_ASEGURADO_IGUAL_POLIZA,
    'RECLAMOS_ANTERIORES' : RECLAMOS_ANTERIORES,
    'EDAD_CONDUCTOR' : EDAD_CONDUCTOR,
    'FECHA_EMISION_ENDOSO_DIAS' : FECHA_EMISION_ENDOSO_DIAS,
    'CODIGO_SUBPRODUCTO' : CODIGO_SUBPRODUCTO,
    'DEPARTAMENTO_SINIESTRO' : DEPARTAMENTO_SINIESTRO,
    'CODIGO_COBERTURA_FRECUENTE' : CODIGO_COBERTURA_FRECUENTE,
    'CANT_COBERTURAS_INVOLUCRADAS' : CANT_COBERTURAS_INVOLUCRADAS,
    'CODIGO_CONCEPTO_RESERVA' : CODIGO_CONCEPTO_RESERVA,
    'ESTADO_SINIESTRO_CALCULADO' : ESTADO_SINIESTRO_CALCULADO,
    'INCURRIDO' : INCURRIDO,
    'FECHA_MOV_MAX_DIAS' : FECHA_MOV_MAX_DIAS,
    'TOMADOR_ASEGURADO_IGUAL_SINIESTRO' : TOMADOR_ASEGURADO_IGUAL_SINIESTRO,
    'TOMADOR_BENEFICIARIO_IGUAL_SINIESTRO' : TOMADOR_BENEFICIARIO_IGUAL_SINIESTRO,
    'ASEGURADO_BENEFICIARIO_IGUAL_SINIESTRO' : ASEGURADO_BENEFICIARIO_IGUAL_SINIESTRO,
    'CODIGO_MUNICIPIO_SINIESTRO' : CODIGO_MUNICIPIO_SINIESTRO,
    'TOTAL_INVOLUCRADOS' : TOTAL_INVOLUCRADOS,
    'DIFF_AVISO_SINIESTRO' : DIFF_AVISO_SINIESTRO,
    'DIFF_INICIO_ENDOSO_SINIESTRO' : DIFF_INICIO_ENDOSO_SINIESTRO,
    'DIFF_FIN_ENDOSO_SINIESTRO' : DIFF_FIN_ENDOSO_SINIESTRO,
    'MUNICIPIO_SINIESTRO_POLIZA_IGUAL' : MUNICIPIO_SINIESTRO_POLIZA_IGUAL,
    'ESTADO_CIVIL' : ESTADO_CIVIL,
    'VINCULACION_LABORAL' : VINCULACION_LABORAL,
    'EGRESOS' : EGRESOS,
    'INGRESOS' : INGRESOS,
    'VALOR_ACTIVO' : VALOR_ACTIVO,
    'VALOR_PASIVO' : VALOR_PASIVO,
    'VALOR_PATRIMONIO' : VALOR_PATRIMONIO,
    'CODIGO_CIIU' : CODIGO_CIIU,
    'ACIERTA_PLUS' : ACIERTA_PLUS,
    'QUANTO' : QUANTO,
    'BONUS_MALUS' : BONUS_MALUS,
    'EDAD_ASEGURADO' : EDAD_ASEGURADO
    }

    new_data = pd.DataFrame(new_data, index=[0]) 
    
    # Se lee lista completa de parametros que deben entrar al SOM
    with open (MODELS_DIR + "listaColumnasDummies.txt", 'rb') as fp:
        list_1 = pickle.load(fp)

    new_data_dummy = pd.get_dummies(new_data)
   
    # Get missing columns in the training test
    missing_cols = set( list_1) - set( new_data_dummy.columns )

    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        new_data_dummy[c] = 0

    # Ensure the order of column in the test set is in the same order than in train set
    new_data_dummy = new_data_dummy[list_1]
    
    
    #Se lee la media y la desviacion estandar del conjunto de entrenamiento
    with open (MODELS_DIR + "MeanModel.txt", 'rb') as fp:
        meanModel = pickle.load(fp)

    with open (MODELS_DIR + "StdModel.txt", 'rb') as fp_2:
        stdModel = pickle.load(fp_2)
    
    # data normalization
    new_data_dummy = (new_data_dummy - meanModel) / stdModel
    new_data_dummy = new_data_dummy.values
    
    #return new_data
    
    return new_data_dummy[0]


def load_model():
    """
        Load Self-Organized map model
    """
    
    with open(MODELS_DIR + 'model_2.p', 'rb') as infile:
        som_2 = pickle.load(infile)

    return som_2


def get_graph():
    """
        Standard function to get a graph that can be displayed in a Django application
    """
    buffer = BytesIO()
    plt.savefig(buffer, format='png')

    buffer.seek(0)
    image_png = buffer.getvalue()

    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')

    buffer.close()
    return graph


def load_frequencies():
    """
        Load pre-processed frequencies
    """
    return pd.read_csv(MODELS_DIR + "frec_2.csv", header=None)


def get_linear_plot(x, y, options = {}):
    """ Linear plot for testing purporse 
        @var x => List of X axis
        @var y => List of Y axis
        @var **options => Custom matplotlib parameters
    """
    plt.switch_backend('AGG')
    plt.figure(figsize=(10,5))
    plt.title('Sales of Items')
    plt.plot(x,y)
    plt.xticks(rotation=45)
    plt.xlabel('item')
    plt.ylabel('price')
    plt.tight_layout
    

    graph = get_graph()
    return graph


def get_seftorganized_plot(options = {}):
    """
        Generates a Seft-Organized chart
    """
    plt.switch_backend('AGG')
    
    if 'figsize' in options:
        f_size = options['figsize']
    else:
        f_size = (7,7)

    if 'cmap' in options:
        colors = options['cmap']
    else:
        colors = 'gray_r'
        
    plt.figure(figsize = f_size)

    if 'title' in options:
        plt.title(options['title'])

    plt.pcolor( options['p_color'] , cmap = colors)  
    
    plt.colorbar()
    plt.tight_layout

    if options['point'] is not None:
        plt.plot(options['point'][0], options['point'][1], 'ro')

    graph = get_graph()
    return graph


def load_scores(score):
    """
        Load pre-processed scores
    """
    df_score = pd.read_csv(MODELS_DIR + "00_RESULTADO_AUX2.csv")
    
    score = float(score)
    df_score = df_score[df_score['PROBABILIDAD'] <= score]

    df_score = df_score[['NUMERO_SECUENCIA_POLIZA',
        'DEPARTAMENTO_RIESGO',
        'DEPARTAMENTO_SINIESTRO',
        'MARCA_VEHICULO',
        'LINEA',
        'MODELO',
        'RECLAMOS_ANTERIORES',
        'VALOR_ASEGURADO_VEHICULO',
        'INCURRIDO',
        'SEXO_CONDUCTOR',
        'EDAD_CONDUCTOR',
        'TOTAL_INVOLUCRADOS',
        'INGRESOS']]
        
    df_size = df_score.shape[0]

    return {'size': df_size, 'data': df_score.to_dict('records')}


def load_soat_kmeans(cluster,city):
    """
        Load pre-processed K-means clustering 
    """
    df_clusters = pd.read_csv(MODELS_DIR + "SINIESTROS_SOAT_02_CLUSTERS.csv")

    df_clusters = df_clusters[df_clusters["MUNICIPIO_SINIESTRO"] == city]
    df_clusters = df_clusters[df_clusters["CLUSTER"] == int(cluster)]

    return df_clusters.to_dict('records')


## Iteration 2 Methods

def get_webapp_parameters(key = None):
    if key is None:
        return Parameter.objects.all()
    
    return Parameter.objects.filter(parameter_key = key)


def getDataForModel_IT2(
        SEXO_ASEGURADO=  np.nan ,
        CODIGO_MUNICIPIO=  np.nan ,
        DEPARTAMENTO_RIESGO=  np.nan ,
        SEXO_CONDUCTOR=  np.nan ,
        NUMERO_ENDOSO=  np.nan ,
        MARCA_VEHICULO=  np.nan ,
        LINEA=  np.nan ,
        MODELO=  np.nan ,
        COD_USO=  np.nan ,
        COD_TIPO=  np.nan ,
        VALOR_ASEGURADO_VEHICULO=  np.nan ,
        VALOR_ASEGURADO_ACCESORIOS=  np.nan ,
        CILINDRADA=  np.nan ,
        TIPO_VEHICULO=  np.nan ,
        ALTURA_POLIZA=  np.nan ,
        CANAL_POLIZA=  np.nan ,
        TOMADOR_ASEGURADO_IGUAL_POLIZA=  np.nan ,
        RECLAMOS_ANTERIORES=  np.nan ,
        EDAD_CONDUCTOR=  np.nan ,
        FECHA_EMISION_ENDOSO_DIAS=  np.nan ,
        CODIGO_SUBPRODUCTO=  np.nan ,
        DEPARTAMENTO_SINIESTRO=  np.nan ,
        CODIGO_CONCEPTO_RESERVA=  np.nan ,
        TOMADOR_ASEGURADO_IGUAL_SINIESTRO=  np.nan ,
        CODIGO_MUNICIPIO_SINIESTRO=  np.nan ,
        TOTAL_INVOLUCRADOS=  np.nan ,
        DIFF_AVISO_SINIESTRO=  np.nan ,
        DIFF_INICIO_ENDOSO_SINIESTRO=  np.nan ,
        DIFF_FIN_ENDOSO_SINIESTRO=  np.nan ,
        MUNICIPIO_SINIESTRO_POLIZA_IGUAL=  np.nan ,
        ESTADO_CIVIL=  np.nan ,
        VINCULACION_LABORAL=  np.nan ,
        EGRESOS=  np.nan ,
        INGRESOS=  np.nan ,
        VALOR_ACTIVO=  np.nan ,
        VALOR_PASIVO=  np.nan ,
        VALOR_PATRIMONIO=  np.nan ,
        CODIGO_CIIU=  np.nan ,
        ACIERTA_PLUS=  np.nan ,
        QUANTO=  np.nan ,
        BONUS_MALUS=  np.nan ,
        EDAD_ASEGURADO=  np.nan ):
    new_data = {
    'SEXO_ASEGURADO' : SEXO_ASEGURADO,
    'CODIGO_MUNICIPIO' : CODIGO_MUNICIPIO,
    'DEPARTAMENTO_RIESGO' : DEPARTAMENTO_RIESGO,
    'SEXO_CONDUCTOR' : SEXO_CONDUCTOR,
    'NUMERO_ENDOSO' : NUMERO_ENDOSO,
    'MARCA_VEHICULO' : MARCA_VEHICULO,
    'LINEA' : LINEA,
    'MODELO' : MODELO,
    'COD_USO' : COD_USO,
    'COD_TIPO' : COD_TIPO,
    'VALOR_ASEGURADO_VEHICULO' : VALOR_ASEGURADO_VEHICULO,
    'VALOR_ASEGURADO_ACCESORIOS' : VALOR_ASEGURADO_ACCESORIOS,
    'CILINDRADA' : CILINDRADA,
    'TIPO_VEHICULO' : TIPO_VEHICULO,
    'ALTURA_POLIZA' : ALTURA_POLIZA,
    'CANAL_POLIZA' : CANAL_POLIZA,
    'TOMADOR_ASEGURADO_IGUAL_POLIZA' : TOMADOR_ASEGURADO_IGUAL_POLIZA,
    'RECLAMOS_ANTERIORES' : RECLAMOS_ANTERIORES,
    'EDAD_CONDUCTOR' : EDAD_CONDUCTOR,
    'FECHA_EMISION_ENDOSO_DIAS' : FECHA_EMISION_ENDOSO_DIAS,
    'CODIGO_SUBPRODUCTO' : CODIGO_SUBPRODUCTO,
    'DEPARTAMENTO_SINIESTRO' : DEPARTAMENTO_SINIESTRO,
    'CODIGO_CONCEPTO_RESERVA' : CODIGO_CONCEPTO_RESERVA,
    'TOMADOR_ASEGURADO_IGUAL_SINIESTRO' : TOMADOR_ASEGURADO_IGUAL_SINIESTRO,
    'CODIGO_MUNICIPIO_SINIESTRO' : CODIGO_MUNICIPIO_SINIESTRO,
    'TOTAL_INVOLUCRADOS' : TOTAL_INVOLUCRADOS,
    'DIFF_AVISO_SINIESTRO' : DIFF_AVISO_SINIESTRO,
    'DIFF_INICIO_ENDOSO_SINIESTRO' : DIFF_INICIO_ENDOSO_SINIESTRO,
    'DIFF_FIN_ENDOSO_SINIESTRO' : DIFF_FIN_ENDOSO_SINIESTRO,
    'MUNICIPIO_SINIESTRO_POLIZA_IGUAL' : MUNICIPIO_SINIESTRO_POLIZA_IGUAL,
    'ESTADO_CIVIL' : ESTADO_CIVIL,
    'VINCULACION_LABORAL' : VINCULACION_LABORAL,
    'EGRESOS' : EGRESOS,
    'INGRESOS' : INGRESOS,
    'VALOR_ACTIVO' : VALOR_ACTIVO,
    'VALOR_PASIVO' : VALOR_PASIVO,
    'VALOR_PATRIMONIO' : VALOR_PATRIMONIO,
    'CODIGO_CIIU' : CODIGO_CIIU,
    'ACIERTA_PLUS' : ACIERTA_PLUS,
    'QUANTO' : QUANTO,
    'BONUS_MALUS' : BONUS_MALUS,
    'EDAD_ASEGURADO' : EDAD_ASEGURADO
    }
    new_data = pd.DataFrame(new_data, index=[0]) 

    # Se lee lista completa de parametros que deben entrar al SOM
    with open (MODELS_DIR + "modelos_it2/listaColumnasDummies.txt", 'rb') as fp:
        list_1 = pickle.load(fp)

    new_data_dummy = pd.get_dummies(new_data)

    # Get missing columns in the training test
    missing_cols = set( list_1) - set( new_data_dummy.columns )

    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        new_data_dummy[c] = 0

    # Ensure the order of column in the test set is in the same order than in train set
    new_data_dummy = new_data_dummy[list_1]


    #Se lee la media y la desviacion estandar del conjunto de entrenamiento
    with open (MODELS_DIR + "modelos_it2/StandarScalerFit", 'rb') as fp:
        standardScaler = pickle.load(fp)

    # data normalization
    new_data_dummy = standardScaler.transform(new_data_dummy)
    
    #return new_data
    return new_data_dummy[0]


def load_model_IT2():
    with open(MODELS_DIR + 'modelos_it2/model_som.p', 'rb') as infile:
        som_front = pickle.load(infile)


    with open(MODELS_DIR + 'modelos_it2/frec_som', 'rb') as infile:
        frec_front = pickle.load(infile)
    
    return som_front, frec_front


def get_3d_freq_plot(options = {}):
    plt.switch_backend('AGG')
    
    df_plot = pd.DataFrame(options['vec'], columns=list('XYZ'))

    if 'figsize' in options:
        f_size = options['figsize']
    else:
        f_size = (7,7)
    
    fig = plt.figure(figsize = f_size)
    ax1 = fig.add_subplot(111, projection='3d')

    x3 = df_plot["X"]
    y3 = df_plot["Y"]
    z3 = np.zeros(df_plot.shape[0])
    
    dx = np.ones(df_plot.shape[0])
    dy = np.ones(df_plot.shape[0])
    dz = df_plot["Z"]

    ax1.bar3d(x3, y3, z3, dx, dy, dz)

    plt.tight_layout

    if options['point'] is not None:
        plt.plot(options['point'][0], options['point'][1], options['max'], 'ro')

    graph = get_graph()
    return graph


def get_threshold_hist_plot(options = {}):
    plt.switch_backend('AGG')

    threshold = float(options['threshold'])

    weight = float(options['weight'])

    
    df_front = load_dataframe_weight()
    df_front["SCORE"] = ((1-df_front["FRECUENCIA_SCALED"]) * weight) + (df_front["QUANT_ERROR_SCALED"] * (1 - weight))

    if 'figsize' in options:
        f_size = options['figsize']
    else:
        f_size = (7,7)

    plt.figure(figsize= f_size)
    plt.hist(df_front["SCORE"], bins= 100)
    plt.axvline(threshold, color='k', linestyle='--')
    plt.xlabel('SCORE')
    plt.ylabel('frequency')

    graph = get_graph()
    return graph


def load_dataframe_weight(complete = True, threshold = 0.88):
    df = pd.read_csv(MODELS_DIR + "modelos_it2/00_RESULTADO_MODELO_COMPLETO.csv", sep = ",")
    if complete:
        return df
    df = df[THRESHOLD_SUMMARY]
    df = df[df["SCORE"] > float(threshold)]
    df = df.round(2)
    df = df.head(1000)
    return df.to_dict('records')
    

def search_client(client_id):
    df_client = pd.read_csv(MODELS_DIR + "QueryClientes.csv")

    df_client = df_client[df_client["KEY_ID_ASEGURADO"] == client_id]
    
    now = datetime.datetime.now().year
    df_client['EDAD_ASEGURADO'] = now - df_client['ANHO']

    df_client.drop(['ANHO'], axis = 1, inplace = True) 

    return df_client.to_dict('records')[0] if df_client.shape[0] == 1 else {}


def all_risk_form_data(post, method='d', dictionary = 'ALLRISK'):
    """
        Transform post data into dict data.
        Parameters:
            Post = request.POST 
            method =    d: Trasnform to pandas notation
                        r: Transform to frontend notation
            reorder = False / True
                Some models are sensible to orden, set True for them.
    """
    data  = {}
    if dictionary == 'ALLRISK':
        structure = ALLRISK_FORM_CONTENT
    elif dictionary == 'SOAT':
        structure = SOAT_FORM_CONTENT
    else:
        structure = {}

    if method == 'd':
        key_list = list(structure.keys())
        val_list = list(structure.values())

        for key, value in post.items():
            if key in val_list:
                position = val_list.index(key)
                data[key_list[position]] = value

        if 'KEY_ID_ASEGURADO' in data:
            data.pop('KEY_ID_ASEGURADO')

    if method == 'r':
        for key, value in post.items():
            front_key = structure[key]
            data[front_key] = value

    return data


# SOAT - ITERACIÓN 2

def load_soat_models():
    """
        Load all related models in k-means in a dictionary
    """
    model_list = {
        'TIPO_DOCUMENTO_TOMADOR'        : MODELS_DIR + 'soat_it2/le_TIPO_DOCUMENTO_TOMADOR.sav',
        'TIPO_DOCUMENTO_ASEGURADO'      : MODELS_DIR + 'soat_it2/le_TIPO_DOCUMENTO_ASEGURADO.sav',
        'MUNICIPIO_SINIESTRO'           : MODELS_DIR + 'soat_it2/le_MUNICIPIO_SINIESTRO.sav',
        'DEPARTAMENTO_SINIESTRO'        : MODELS_DIR + 'soat_it2/le_DEPARTAMENTO_SINIESTRO.sav',
        'SEXO_ASEGURADO'                : MODELS_DIR + 'soat_it2/le_SEXO_ASEGURADO.sav',
        'NOMBRE_MUNICIPIO_MOVILIZACIN'  : MODELS_DIR + 'soat_it2/le_NOMBRE_MUNICIPIO_MOVILIZACIN.sav',
        'NOMBRE_DEPARTAMEN_MOVILIZACIN' : MODELS_DIR + 'soat_it2/le_NOMBRE_DEPARTAMEN_MOVILIZACIN.sav',
        'MARCA'                         : MODELS_DIR + 'soat_it2/le_MARCA.sav',
        'VINCULACION_LABORAL'           : MODELS_DIR + 'soat_it2/le_VINCULACION_LABORAL.sav',
        'ESTRATO'                       : MODELS_DIR + 'soat_it2/le_ESTRATO.sav',
        'CARGO'                         : MODELS_DIR + 'soat_it2/le_CARGO.sav',
        'EMPRESA_TRABAJA'               : MODELS_DIR + 'soat_it2/le_EMPRESA_TRABAJA.sav',
        'ACTIVIDAD_ECONOMICA'           : MODELS_DIR + 'soat_it2/le_ACTIVIDAD_ECONOMICA.sav',
        'SEXO'                          : MODELS_DIR + 'soat_it2/le_SEXO.sav',
        'ESTADO_CIVIL'                  : MODELS_DIR + 'soat_it2/le_ESTADO_CIVIL.sav',
        'KMEANS'                        : MODELS_DIR + 'soat_it2/model_kmeans.sav',
        'SCALER'                        : MODELS_DIR + 'soat_it2/scaler.sav'
    }

    models = {}

    for key, value in model_list.items():
        models[key] = (pickle.load(open(value, 'rb')))

    return models


def soat_transforms(data):
    data = parse_dictionay(data,'SOAT')

    df_models = pd.DataFrame(data, index=[0])

    models = load_soat_models()


    for key, value in models.items():
        if key == 'KMEANS' or key == 'SCALER':
            continue
        df_models[key] = value.transform(df_models[key])

    df_train = models['SCALER'].transform(df_models)
    label = models['KMEANS'].predict(df_train)

    first = df_train
    second = models['KMEANS'].cluster_centers_[label]
    distance = np.linalg.norm(first - second)

    return distance


def parse_dictionay(data, method):
    
    if method == 'SOAT':
        for key, value in data.items():
            data[key] = SOAT_PARSE[key](value)

    return data


def reorder(data,model):
    result = {}

    if model == 'SOAT':
        structure = SOAT_FORM_CONTENT
    
    for key, value in structure.items():
        result[key] = data[key]
    
    return result


def calculate_soat_risk(distance):

    quartiles = get_webapp_parameters('QUARTIL')
    
    start = 0
    for row in quartiles:
        end = float(row.parameter_verbose)
        if distance >= start and distance <= end:
            quatile = row.parameter_value
        start = end
    
    return SOAT_QUARTILE_VERBOSE[quatile]


def load_dataframe_clusters(complete = True):
    df = pd.read_csv(MODELS_DIR + "soat_it2/SiniestrosSoatRevisar.csv", sep = ",")
    if complete:
        return df
    df = df[CLUSTER_SUMMARY]
    df = df.round(2)
    return df.to_dict('records')
    