# Generated by Django 4.2.11 on 2024-03-07 14:55

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('callmodel', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='totalload',
            name='date',
            field=models.DateTimeField(default=datetime.datetime.now),
        ),
        migrations.AddField(
            model_name='totalload',
            name='totalload',
            field=models.DecimalField(decimal_places=1, default=0.0, max_digits=6),
            preserve_default=False,
        ),
    ]