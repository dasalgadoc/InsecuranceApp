ALLRISK_FORM_CONTENT = {
    'KEY_ID_ASEGURADO'                  :   'id_cliente',
    'CODIGO_CIIU'	                    :	'city_insured',
    'SEXO_ASEGURADO'	                :	'sex_insured',
    'EDAD_ASEGURADO'	                :	'age_insured',
    'ESTADO_CIVIL'	                    :	'civil_insured',
    'VINCULACION_LABORAL'	            :	'job_insured',
    'BONUS_MALUS'	                    :	'bm_insured',
    'INGRESOS'	                        :	'income_insured',
    'EGRESOS'	                        :	'expenses_insured',
    'VALOR_ACTIVO'	                    :	'assets_insured',
    'VALOR_PASIVO'	                    :	'passives_insured',
    'VALOR_PATRIMONIO'	                :	'heritage_insured',
    'ACIERTA_PLUS'	                    :	'acierta_insured',
    'QUANTO'	                        :	'quanto_insured',
    'DEPARTAMENTO_RIESGO'	            :	'state_policy',
    'CODIGO_MUNICIPIO'	                :	'city_policy',
    'NUMERO_ENDOSO'	                    :	'end_policy',
    'VALOR_ASEGURADO_VEHICULO'	        :	'amount_policy',
    'VALOR_ASEGURADO_ACCESORIOS'	    :	'amount_a_policy',
    'TOMADOR_ASEGURADO_IGUAL_POLIZA'	:	'taken_policy',
    'FECHA_EMISION_ENDOSO_DIAS'	        :	'em_date_policy',
    'COD_USO'	                        :	'use_policy',
    'COD_TIPO'	                        :	'type_policy',
    'ALTURA_POLIZA'	                    :	'height_policy',
    'CANAL_POLIZA'	                    :	'channel_policy',
    'CODIGO_SUBPRODUCTO'	            :	'subproduct_policy',
    'CODIGO_CONCEPTO_RESERVA'	        :	'covert_policy',
    'DEPARTAMENTO_SINIESTRO'	        :	'state_sinister',
    'CODIGO_MUNICIPIO_SINIESTRO'	    :	'city_sinister',
    'SEXO_CONDUCTOR'	                :	'gender_sinister',
    'EDAD_CONDUCTOR'	                :	'age_driver',
    'RECLAMOS_ANTERIORES'	            :	'claims_sinister',
    'TOTAL_INVOLUCRADOS'	            :	'involved_sinister',
    'TOMADOR_ASEGURADO_IGUAL_SINIESTRO'	:	'taker_a_sinister',
    'MUNICIPIO_SINIESTRO_POLIZA_IGUAL'	:	'town_sinister',
    'DIFF_AVISO_SINIESTRO'	            :	'annon_sinister',
    'DIFF_INICIO_ENDOSO_SINIESTRO'	    :	'start_sinister',
    'DIFF_FIN_ENDOSO_SINIESTRO'	        :	'end_sinister',
    'TIPO_VEHICULO'	                    :	'type_vehicle',
    'MARCA_VEHICULO'	                :	'brand_vehicle',
    'LINEA'	                            :	'line_vehicle',
    'MODELO'	                        :	'model_vehicle',
    'CILINDRADA'	                    :	'cylinder_vehicle',
}

SOAT_FORM_CONTENT = {
    'CODIGO_RIESGO'						:	'risk_policy',
    'TIPO_DOCUMENTO_TOMADOR'			:	'document_policy',
    'TIPO_DOCUMENTO_ASEGURADO'			:	'document_insured',
    'MUNICIPIO_SINIESTRO'				:	'city_sinister',
    'DEPARTAMENTO_SINIESTRO'			:	'state_sinister',
    'DIFERENCIA_AVISO_SINIESTRO'		:	'difano_sinister',
    'SEXO_ASEGURADO'					:	'sex_insured',
    'NOMBRE_MUNICIPIO_MOVILIZACIN'		:	'city_policy',
    'NOMBRE_DEPARTAMEN_MOVILIZACIN'		:	'state_policy',
    'NUMERO_ENDOSO'						:	'end_policy',
    'MODELO'							:	'model_vehicle',
    'VALOR_ASEGURADO_VEHICULO'			:	'amount_policy',
    'MARCA'								:	'brand_vehicle',
    'EDAD_ASEGURADO'					:	'age_insured',
    'VINCULACION_LABORAL'				:	'job_insured',
    'ESTRATO'							:	'stratum_insured',
    'CARGO'								:	'charge_insured',
    'EMPRESA_TRABAJA'					:	'enterprise_insured',
    'VALOR_EGRESOS'						:	'expenses_insured',
    'VALOR_INGRESOS'					:	'income_insured',
    'VALOR_ACTIVO'						:	'assets_insured',
    'VALOR_PASIVO'						:	'passives_insured',
    'VALOR_PATRIMONIO'					:	'heritage_insured',
    'CODIGO_CIIU'						:	'city_insured',
    'ACTIVIDAD_ECONOMICA'				:	'activity_insured',
    'SEXO'								:	'sex_policy',
    'ESTADO_CIVIL'						:	'civil_insured',
    'RENOVACIONES'						:	'revenue_policy',
    'MISMO_MUNICIPIO_POLIZA_SINIESTRO'	:	'citypo_sinister',
    'DIFERENCIA_SINIESTRO_ENDOSO'		:	'difend_sinister',
    'CANTIDAD_COBERTURAS'				:	'covert_policy',
    'CANTIDAD_RESERVAS'					:	'reserv_policy',
    'CANTIDAD_SINIESTROS'				:	'sinister_policy',
    'MISMO_TOMADOR_ASEGURADO'			:	'taken_insured',
    'RELACION_VALOR_INGRESOS'			:	'related_policy'
}

SOAT_PARSE = {
    'CODIGO_CIIU'						:	int,
    'SEXO_ASEGURADO'					:	str,
    'EDAD_ASEGURADO'					:	int,
    'TIPO_DOCUMENTO_ASEGURADO'			:	str,
    'ESTRATO'							:	int,
    'ESTADO_CIVIL'						:	str,
    'ACTIVIDAD_ECONOMICA'				:	str,
    'EMPRESA_TRABAJA'					:	str,
    'VINCULACION_LABORAL'				:	str,
    'CARGO'								:	str,
    'MISMO_TOMADOR_ASEGURADO'			:	int,
    'VALOR_EGRESOS'						:	int,
    'VALOR_INGRESOS'					:	int,
    'VALOR_ACTIVO'						:	int,
    'VALOR_PASIVO'						:	int,
    'VALOR_PATRIMONIO'					:	int,
    'NOMBRE_DEPARTAMEN_MOVILIZACIN'		:	str,
    'NOMBRE_MUNICIPIO_MOVILIZACIN'		:	str,
    'NUMERO_ENDOSO'						:	int,
    'CODIGO_RIESGO'						:	int,
    'TIPO_DOCUMENTO_TOMADOR'			:	str,
    'SEXO'								:	str,
    'VALOR_ASEGURADO_VEHICULO'			:	int,
    'CANTIDAD_COBERTURAS'				:	int,
    'CANTIDAD_RESERVAS'					:	int,
    'CANTIDAD_SINIESTROS'				:	int,
    'RELACION_VALOR_INGRESOS'			:	int,
    'RENOVACIONES'						:	int,
    'DEPARTAMENTO_SINIESTRO'			:	str,
    'MUNICIPIO_SINIESTRO'				:	str,
    'DIFERENCIA_SINIESTRO_ENDOSO'		:	int,
    'DIFERENCIA_AVISO_SINIESTRO'		:	int,
    'MISMO_MUNICIPIO_POLIZA_SINIESTRO'	:	int,
    'MODELO'							:	int,
    'MARCA'								:	str
}

THRESHOLD_SUMMARY = [
    'NUMERO_SECUENCIA_POLIZA',
    'CODIGO_RIESGO',
    'MARCA_VEHICULO',
    'CODIGO_SUBPRODUCTO',
    'DEPARTAMENTO_SINIESTRO',
    'TOTAL_INVOLUCRADOS',
    'X',
    'Y',
    'FRECUENCIA',
    'PORCENTAJE_FREC',
    'DISTANCIA',
    'CLUSTER_SOM',
    'IS_OUTLIER',
    'FRECUENCIA_SCALED',
    'SCORE'
]

CLUSTER_SUMMARY = [
    'NUMERO_SECUENCIA_POLIZA',
    'NUMERO_POLIZA',
    'NUMERO_SINIESTRO',
    'TIPO_DOCUMENTO_TOMADOR',
    'TIPO_DOCUMENTO_ASEGURADO',
    'MUNICIPIO_SINIESTRO',
    'DEPARTAMENTO_SINIESTRO',
    'FECHA_SINIESTRO',
    'FECHA_AVISO',
    'CODIGO_COBERTURA',
    'CODIGO_CONCEPTO_RESERVA',
    'FECHA_MOV_MAX',
    'INCURRIDO',
    'LIQUIDADO',
    'KmeansDistance'
]

CLIENT_NO_FOUND = 'El cliente ingresado no éxite en la base de datos.'
CLIENT_FOUND = 'Cliente localizado en la base de datos'
POINT_CALCULATED = 'La información ingresada ubicaría el punto en la coordenada {}.'
SOAT_DISTANCE = 'La distancia calculada para este siniestro es {}.'
SOAT_RISK = 'Este siniestro se considera {}.'

SOAT_QUARTILE_VERBOSE = {
    'Q1' : 'Riesgo muy bajo',
    'Q2' : 'Riesgo bajo',
    'Q3' : 'Riesgo moderado',
    'Q4' : 'Riesgo alto'
}