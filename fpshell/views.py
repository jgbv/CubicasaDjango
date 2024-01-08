from django.shortcuts import render
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import FPCCImage
from .serializers import ImageSerializer
from PIL import Image as PILImage
from . import FPCCParse as fpcc

import os

def index(request):
    return render(request, 'homepage.html')

class FPCCShellUpload(APIView):
    parser_classes = (FileUploadParser,)
    fpccImgfolder = "fpccshellimages"
    def post(self, request, *args, **kwargs):
        print("====FPCCShellPOST====")
        file_serializer = ImageSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            uploaded_image = file_serializer.instance
            with PILImage.open(uploaded_image.file) as img:
                # image_dimensions = img.size
                # imgX = img.size[0]
                # imgY = img.size[1]
                savepath = os.path.join("fpshell", FPCCShellUpload.fpccImgfolder,f"{request.data['file'].name}")
                img.save(savepath)
            # response_data = pfcv.RBGFloorPlanOpenCV.getOuterShell(img_path=savepath)
            response_data = fpcc.CCInference.makeInference(savepath, FPCCShellUpload.fpccImgfolder)
            print(f"!-=-=-=-=-=-=--=-=-response_data: {response_data}")
            
            response = Response(response_data, status=status.HTTP_201_CREATED)
            print("-------------response: {response}------------")
            # response['Content-Disposition'] = f'attachment; filename={response_data["imagePath"]}'
            # return Response(response_data, status=status.HTTP_201_CREATED)
            FPCCShellUpload.clear_images_folder('media/fpccshellimages')
            return response
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    
    def clear_images_folder(folder_path):
        # folder_path = 'media/fpccimages'
        print(f"!------{folder_path} exists?: {os.path.exists(folder_path)}")
        if os.path.exists(folder_path):
            for f in os.scandir(folder_path):
                if not f.is_dir():
                    imgpath=f.path
                    try:
                        os.remove(f.path)
                        print(f"removed {imgpath}")
                    except Exception as e:
                        print(f"Error removing {f.path}: {e}")
        else:
            print("Images folder does not exist.")
    
