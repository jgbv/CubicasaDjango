from rest_framework import serializers
from .models import FPCCImage

# print("IN SERIALIZERS")
# print(Image)

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = FPCCImage
        fields = '__all__'