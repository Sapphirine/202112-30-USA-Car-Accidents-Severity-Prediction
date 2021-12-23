from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from . import predict

def homepage(request):
    context = {}
    context['homepage'] = "home page"
    return render(request, "index.html", context)

def prediction(request):
    lon = float(request.GET.get('lon'))
    lat = float(request.GET.get('lat'))
    model = predict.Backend()
    data = model.predict([lon,lat])
    
    return JsonResponse(data)
    