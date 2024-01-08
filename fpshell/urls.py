from django.urls import path
from .views import FPCCShellUpload

urlpatterns = [
    path('fpccshellupload/', FPCCShellUpload.as_view(), name='fpccshellupload'),
    path('', index, name="index")
]
