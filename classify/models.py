from django.db import models


class Image(models.Model):
    id = models.AutoField(primary_key=True)
    path = models.CharField(max_length=255)
    predicted_label = models.CharField(max_length=100)

    def __str__(self):
        return self.path