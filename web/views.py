from django.shortcuts import render
from .predictor import obtener_datos_y_prediccion

def vista_prediccion(request):

    datos_60 = None
    resultado = None

    # Cargar datos siempre (los 60 días)
    datos_60, prediccion = obtener_datos_y_prediccion()

    # Solo mostrar predicción cuando el usuario presiona "Predecir"
    if request.method == "POST":
        if "btn_predecir" in request.POST:
            resultado = prediccion

        if "btn_limpiar" in request.POST:
            resultado = None  # no mostrar nada

    contexto = {
        "datos_60": datos_60,
        "resultado": resultado,
    }

    return render(request, "prediccion.html", contexto)
