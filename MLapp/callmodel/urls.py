from django.urls import path
from  . import views

urlpatterns = [
    path("", views.createchart,name="index"),
    #path("", views.scheduleAPI,name="schedule")
]