from django.urls import path
from . import views

urlpatterns=[
    
    path('',views.home,name='home'),
    path('send/',views.chat1,name='chat1'),
    path('chat1/',views.chat1,name='chat1'),
    path('search/',views.search,name='search'),
    path('check-out/',views.checkout,name='check-out'),
    path('faq/',views.faq,name='faq'),
    path('check-out/delete/',views.delete1,name='delete'),
    path('shopping-cart/',views.cart,name='shopping-cart'),
]
