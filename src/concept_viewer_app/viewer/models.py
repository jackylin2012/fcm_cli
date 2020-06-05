from django.db import models

# Create your models here.


class Result(models.Model):
    key = models.CharField(max_length=128, primary_key=True)
    dataset = models.CharField(max_length=64)
    grid_dir = models.CharField(max_length=64)
    run_id = models.CharField(max_length=64)
    window_size = models.IntegerField()
    embed_dim = models.IntegerField()
    nnegs = models.IntegerField()
    nconcepts = models.IntegerField()
    lam = models.FloatField()
    rho = models.FloatField()
    eta = models.FloatField()
    topics = models.TextField(default='')

    def __str__(self):
        return '{}:rho{}'.format(self.run_id, self.rho)
