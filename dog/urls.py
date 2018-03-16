from django.conf.urls import url

import view
import ml
import settings
urlpatterns = [
    url(r'^hello$', view.hello),
    url(r'^show$', view.show),
    url(r'^$', view.index),
    url(r'^search$', ml.search),
    url(r'^updateInfo$',  ml.updateInfo),
    url(r'^registersuccess$', view.save),
    url(r'^register$', view.register),
    url(r'^signin$', view.signin),
    url(r'^userlogin$', view.userlogin),
]

