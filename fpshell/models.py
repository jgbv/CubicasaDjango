from django.db import models


class FPCCImage(models.Model):
    file = models.ImageField(upload_to='fpccshellimages/')
    uploaded_at = models.DateTimeField(auto_now_add=True)