""" 
    main_app module url management
"""

from django.urls import path
from main_app import views

urlpatterns =  [
    path(
        route='',
        view=views.home,
        name='home'
    ),

    path(
        route='1',
        view=views.all_risk_1,
        name='1'
    ),

    path(
        route='2',
        view=views.all_risk_2,
        name='2'
    ),

    path(
        route='3',
        view=views.all_risk_3,
        name='3'
    ),

    path(
        route='4',
        view=views.all_risk_4,
        name='4'
    ),

    path(
        route='s1',
        view=views.soat_1,
        name='s1'
    ),

    path(
        route='s2',
        view=views.soat_2,
        name='s2'
    ),

]