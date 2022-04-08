# Django libraries
from typing import NewType
from django.shortcuts import render, resolve_url

# Utils functions
from .utils import getDataForModel 
from .utils import load_model
from .utils import get_linear_plot
from .utils import get_seftorganized_plot
from .utils import load_frequencies
from .utils import load_scores
from .utils import load_soat_kmeans
from .utils import get_webapp_parameters
from .utils import load_model_IT2
from .utils import get_3d_freq_plot
from .utils import search_client
from .utils import all_risk_form_data
from .utils import getDataForModel_IT2
from .utils import get_threshold_hist_plot
from .utils import load_dataframe_weight
from .utils import soat_transforms
from .utils import reorder
from .utils import calculate_soat_risk
from .utils import load_dataframe_clusters

from .data_structures import CLIENT_NO_FOUND
from .data_structures import CLIENT_FOUND
from .data_structures import POINT_CALCULATED
from .data_structures import SOAT_DISTANCE
from .data_structures import SOAT_RISK

def home(request):
    """
        View to generate HTML Template respose to URL <IP>:<PORT>/
        @var request: HTTML REQUEST
    """
    return render(request, 'main_app/index.html')


def all_risk_1(request):
    """
        View to generate HTML Template respose to URL <IP>:<PORT>/todo_riesgo/1
        @var request: HTTML REQUEST
    """
    som_2 = load_model()

    data = {}
    point = None

    if request.method == 'POST':
        test_val = getDataForModel(
            SEXO_ASEGURADO  = request.POST['sex_insured'],
            CODIGO_MUNICIPIO  = request.POST['city_policy'],
            DEPARTAMENTO_RIESGO  = request.POST['state_policy'],
            SEXO_CONDUCTOR  = request.POST['gender_sinister'],
            NUMERO_ENDOSO  = request.POST['code_policy'],
            MARCA_VEHICULO  = request.POST['brand_vehicle'],
            LINEA  = request.POST['line_vehicle'],
            MODELO  = request.POST['model_vehicle'],
            COD_USO  = request.POST['use_policy'],
            COD_TIPO  = request.POST['type_policy'],
            VALOR_ASEGURADO_VEHICULO  = request.POST['amount_policy'],
            VALOR_ASEGURADO_ACCESORIOS  = request.POST['amount_a_policy'],
            CILINDRADA  = request.POST['cylinder_vehicle'],
            TIPO_VEHICULO  = request.POST['type_vehicle'],
            ALTURA_POLIZA  = request.POST['height_policy'],
            CANAL_POLIZA  = request.POST['channel_policy'],
            TOMADOR_ASEGURADO_IGUAL_POLIZA  = request.POST['taken_policy'],
            RECLAMOS_ANTERIORES  = request.POST['claims_sinister'],
            EDAD_CONDUCTOR  = request.POST['age_sinister'],
            FECHA_EMISION_ENDOSO_DIAS  = request.POST['date_policy'],
            CODIGO_SUBPRODUCTO  = request.POST['subproduct_policy'],
            DEPARTAMENTO_SINIESTRO  = request.POST['state_sinister'],
            CODIGO_COBERTURA_FRECUENTE  = request.POST['covert_policy'],
            CANT_COBERTURAS_INVOLUCRADAS  = request.POST['involved_policy'],
            CODIGO_CONCEPTO_RESERVA  = request.POST['concept_policy'],
            ESTADO_SINIESTRO_CALCULADO  = request.POST['stage_sinister'],
            INCURRIDO  = request.POST['incurred_sinister'],
            FECHA_MOV_MAX_DIAS  = request.POST['date_sinister'],
            TOMADOR_ASEGURADO_IGUAL_SINIESTRO  = request.POST['taker_a_sinister'],
            TOMADOR_BENEFICIARIO_IGUAL_SINIESTRO  = request.POST['taker_b_sinister'],
            ASEGURADO_BENEFICIARIO_IGUAL_SINIESTRO  = request.POST['insured_sinister'],
            CODIGO_MUNICIPIO_SINIESTRO  = request.POST['city_sinister'],
            TOTAL_INVOLUCRADOS  = request.POST['involved_sinister'],
            DIFF_AVISO_SINIESTRO  = request.POST['annon_sinister'],
            DIFF_INICIO_ENDOSO_SINIESTRO  = request.POST['start_sinister'],
            DIFF_FIN_ENDOSO_SINIESTRO  = request.POST['end_sinister'],
            MUNICIPIO_SINIESTRO_POLIZA_IGUAL  = request.POST['town_sinister'],
            ESTADO_CIVIL  = request.POST['civil_insured'],
            VINCULACION_LABORAL  = request.POST['job_insured'],
            EGRESOS  = request.POST['expenses_insured'],
            INGRESOS  = request.POST['income_insured'],
            VALOR_ACTIVO  = request.POST['assets_insured'],
            VALOR_PASIVO  = request.POST['passives_insured'],
            VALOR_PATRIMONIO  = request.POST['heritage_insured'],
            CODIGO_CIIU  = request.POST['city_insured'],
            ACIERTA_PLUS  = request.POST['acierta_insured'],
            QUANTO  = request.POST['quanto_insured'],
            BONUS_MALUS  = request.POST['bm_insured'],
            EDAD_ASEGURADO  = request.POST['age_insured']
        )
        
        point = data['message'] = som_2.winner(test_val)
        
            # (29, 11)

        
    options = {
        'figsize':(7,7),
        'cmap': 'gray_r',
        'p_color': som_2.distance_map().T,
        'point': point 
    }

    chart = get_seftorganized_plot(options)
    data['chart'] = chart

    frequencies = load_frequencies()

    options = {
        'figsize':(7,7),
        'cmap': 'Blues',
        'p_color': frequencies.T,
        'point': point
    }

    chart_2 = get_seftorganized_plot(options)
    data['chart_2'] = chart_2

    return render(request, 'main_app/all_risk_1.html', data)


def all_risk_2(request):
    """
        View to generate HTML Template respose to URL <IP>:<PORT>/todo_riesgo/2
        @var request: HTTML REQUEST
    """
    if request.method == 'POST':
        scores = load_scores(request.POST['score'])
        
        return render(request, 'main_app/all_risk_2.html', {'scores' : scores['size'], 'data': scores['data'] })
    
    return render(request, 'main_app/all_risk_2.html')


def all_risk_3(request):
    """
        View to generate HTML Template respose to URL <IP>:<PORT>/todo_riesgo/3
        @var request: HTTML REQUEST
    """
    data = {'selects': get_webapp_parameters()}

    som_2 , frequencies = load_model_IT2()
    point = None
    threshold = data['threshold'] = 0.88
    weight = data['weight'] = 0.7

    if request.method == 'POST':
        if 'fillInfo' in request.POST or 'locatePoint' in request.POST:
            data['data_filled'] = 'yes'
            data['form_data'] = request.POST.copy()

            if 'fillInfo' in request.POST:
                print("Rellenar datos")
                
                client = {}

                if request.POST['id_cliente'] != '':
                    client = search_client(request.POST['id_cliente'])
                    if len(client) > 0:
                        front_client = all_risk_form_data(post = client, method='r')
                        for key, value in front_client.items():
                            data['form_data'][key] = value
                    
                
                if len(client) == 0:
                    data['message'] = CLIENT_NO_FOUND
                else:
                    data['message'] = CLIENT_FOUND
          

            if 'locatePoint' in request.POST:
                print("Localizar datos")
                pandas_form = all_risk_form_data(post = data['form_data'])
                toggles = ['TOMADOR_ASEGURADO_IGUAL_POLIZA','TOMADOR_ASEGURADO_IGUAL_SINIESTRO','MUNICIPIO_SINIESTRO_POLIZA_IGUAL']
                
                if 'SEXO_CONDUCTOR' in pandas_form:
                    pandas_form['SEXO_CONDUCTOR'] = 'M'
                else:
                    pandas_form['SEXO_CONDUCTOR'] = 'F'

                for toggle in toggles:
                    if toggle in pandas_form:
                        pandas_form[toggle] = '1'
                    else:
                        pandas_form[toggle] = '0'

                array_result = getDataForModel_IT2(**pandas_form)
                point = som_2.winner(array_result)
                data['message'] = POINT_CALCULATED.format(point)
            

        if 'thr_wei_button' in request.POST:
            print("Calcular umbral y peso")
            #for key, value in request.POST.items():
            #    print(key, '->', value)
            threshold = request.POST['threshold']
            weight = request.POST['weight_range']


    options = {
        'figsize':(7,7),
        'cmap': 'Purples',
        'p_color': som_2.distance_map().T,
        'point': point 
    }

    chart = get_seftorganized_plot(options)
    data['chart'] = chart

    options = {
        'figsize':(7,7),
        'cmap': 'Reds',
        'p_color': frequencies.T,
        'point': point
    }

    chart_2 = get_seftorganized_plot(options)
    data['chart_2'] = chart_2

    countX = 0
    countY = 0
    vec = []
    max = 0
    for x in frequencies:
        countY = 0
        for y in frequencies:
            vec.append([countX,countY,frequencies[countX][countY]])
            if max < frequencies[countX][countY]:
                max = frequencies[countX][countY]
            countY += 1
        countX += 1

    options = {
        'figsize':(7,7),
        'vec': vec,
        'point': point,
        'max': max
    }
    chart_3 = get_3d_freq_plot(options)
    data['chart_3'] = chart_3

    options = {
        'figsize':(7,7),
        'threshold': threshold,
        'weight': weight
    }
    chart_4 = get_threshold_hist_plot(options)
    data['chart_4'] = chart_4

    data['threshold'] = threshold
    data['weight'] = weight 
    data['thr_table'] = load_dataframe_weight(complete= False, threshold= threshold)

    return render(request, 'main_app/all_risk_3.html', data)


def all_risk_4(request):
    """
        View to generate HTML Template respose to URL <IP>:<PORT>/todo_riesgo/4
        @var request: HTTML REQUEST
    """
    data = {'selects': get_webapp_parameters()}
    if request.method == 'POST':
        if 'fillInfo' in request.POST or 'locatePoint' in request.POST:
            data['data_filled'] = 'yes'
            data['form_data'] = request.POST.copy()

            if 'fillInfo' in request.POST:
                print("Rellenar datos")
                
                client = {}

                if request.POST['id_cliente'] != '':
                    client = search_client(request.POST['id_cliente'])
                    if len(client) > 0:
                        front_client = all_risk_form_data(post = client, method='r')
                        for key, value in front_client.items():
                            data['form_data'][key] = value
                    
                
                if len(client) == 0:
                    data['message'] = CLIENT_NO_FOUND
                else:
                    data['message'] = CLIENT_FOUND
          

            if 'locatePoint' in request.POST:
                som_2 , frequencies = load_model_IT2()
                print("Localizar datos")
                pandas_form = all_risk_form_data(post = data['form_data'])
                toggles = ['TOMADOR_ASEGURADO_IGUAL_POLIZA','TOMADOR_ASEGURADO_IGUAL_SINIESTRO','MUNICIPIO_SINIESTRO_POLIZA_IGUAL']
                
                if 'SEXO_CONDUCTOR' in pandas_form:
                    pandas_form['SEXO_CONDUCTOR'] = 'M'
                else:
                    pandas_form['SEXO_CONDUCTOR'] = 'F'

                for toggle in toggles:
                    if toggle in pandas_form:
                        pandas_form[toggle] = '1'
                    else:
                        pandas_form[toggle] = '0'

                array_result = getDataForModel_IT2(**pandas_form)
                point = som_2.winner(array_result)
                data['message'] = POINT_CALCULATED.format(point)
            
    return render(request, 'main_app/all_risk_4.html', data)


def soat_1(request):
    """
        View to generate HTML Template respose to URL <IP>:<PORT>/soat/s1
        @var request: HTTML REQUEST
    """
    if request.method == 'POST':
        clusters = load_soat_kmeans(request.POST['cluster'],request.POST['country_code'])
        return render(request, 'main_app/soat_1.html', { 'data': clusters })

    return render(request, 'main_app/soat_1.html')


def soat_2(request):
    data = {
        'selects': get_webapp_parameters(),
        'cluster_table' : load_dataframe_clusters(False)
    }
    if request.method == "POST":
        data['data_filled'] = 'yes'
        data['form_data'] = request.POST.copy()
        pandas_form = all_risk_form_data(post = data['form_data'], dictionary='SOAT')
        toggles = ['MISMO_TOMADOR_ASEGURADO','MISMO_MUNICIPIO_POLIZA_SINIESTRO']

        for toggle in toggles:
            if toggle in pandas_form:
                pandas_form[toggle] = '1'
            else:
                pandas_form[toggle] = '0'

        pandas_form = reorder(pandas_form,model='SOAT')

        distance = soat_transforms(pandas_form)

        risk = calculate_soat_risk(distance)

        data['message'] = [SOAT_DISTANCE.format(distance) , SOAT_RISK.format(risk)]

    return render(request, 'main_app/soat_2.html', data)


