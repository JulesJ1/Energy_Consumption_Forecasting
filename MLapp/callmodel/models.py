from django.db import models
from datetime import datetime

# python manage.py makemigrations
# python manage.py migrate
#celery -A project worker -1 INFO
#celery -A MLapp.celery worker -l info


class Totalload(models.Model):
    date = models.DateTimeField(default=datetime.now)
    totalload = models.DecimalField(max_digits = 6, 
                         decimal_places = 1)

    def __str__(self) -> str:
        return str(self.date)

