# Create your views here.
from django.core.serializers import serialize
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.http import require_http_methods

from .models import Result
from .process import process


@require_http_methods(["GET"])
def get_result(request):
    # https://docs.djangoproject.com/en/2.0/topics/serialization/
    results = Result.objects.all()
    response = serialize('json', results)
    return HttpResponse(response)


@require_http_methods(["GET"])
def index(request):
    results = Result.objects.all()
    template = loader.get_template('viewer/index.html')
    context = {
        'results': results,
    }
    return HttpResponse(template.render(context, request))


@require_http_methods(["GET"])
def update_table(request):
    mindf = float(request.GET.get('min_df', 0.01))
    maxdf = float(request.GET.get('max_df', 1.))
    mintf = float(request.GET.get('min_tf', 0.))
    maxtf = float(request.GET.get('max_tf', 1.))
    frex = float(request.GET.get('frex', 0.5))
    results = Result.objects.all()
    handler = process(mindf, maxdf, mintf, maxtf, frex, results[0].dataset)
    results = map(handler, results)
    template = loader.get_template('viewer/table.html')
    context = {
        'results': results,
    }
    return HttpResponse(template.render(context, request))
