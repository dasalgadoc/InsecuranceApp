from django.db import models

# Create your models here.

class Parameter(models.Model):
    """
        Class of parámeters
    """
    parameter_key = models.CharField(max_length = 20, null = False, blank = False, verbose_name = 'Llave Parametro')
    parameter_value = models.CharField(max_length = 50, null = False, blank = False, verbose_name = 'Valor Parametro')
    parameter_verbose = models.CharField(max_length = 50, null = False, blank = False, verbose_name = 'Valor para Mostrar')


    class Meta():
        """
            Parameter subclass to set meta information in Parameter
        """
        verbose_name = 'Parametro'
        verbose_name_plural = 'Parametros'

    
    def __str__(self):
        """
            ToString method from Parameter Class
        """
        return ("{} -> {}".format(self.parameter_key, self.parameter_value))


