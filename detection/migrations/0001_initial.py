# Generated by Django 3.2.21 on 2023-10-04 09:55

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ASLMODEL',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('path', models.CharField(max_length=200)),
                ('created_on', models.DateTimeField(default=django.utils.timezone.now)),
                ('active', models.BooleanField()),
            ],
        ),
    ]
