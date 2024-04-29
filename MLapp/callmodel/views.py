from django.shortcuts import render
from django.http import HttpResponse
from django_celery_beat.models import PeriodicTask, IntervalSchedule
from json import dumps
#from .tasks import callAPI
from .tasks import my_task
import plotly.express as px
from callmodel.models import Totalload
# Create your views here.

def index(request):
    my_task.delay()
    return HttpResponse("Task Started")

def Welcome(request):
    return HttpResponse("Hello World")

def scheduleAPI(requests):
    interval,_ = IntervalSchedule.objects.get_or_create(
        every=2,
        period= IntervalSchedule.MINUTES
    )
    PeriodicTask.objects.create(
        interval=interval,
        name="energy-api-schedule",
        task="callmodel.tasks.my_task",
    
    
    )
    
    return HttpResponse("task scheduled")

def createchart(request):
    df = Totalload.objects.all().order_by("date")
    fig = px.line(
        x = [c.date for c in df],
        y = [c.totalload for c in df],
        title= "Total Load For Last 12 Hours",
        labels={"x":"Time","y":"Total Load"}

    )
    chart = fig.to_html()
    context = {"chart": chart}
    return render(request,"main.html",context)