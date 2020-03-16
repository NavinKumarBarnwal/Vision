from django.contrib import admin
from .models import Product,UserProduct

admin.site.register(Product)
admin.site.register(UserProduct)