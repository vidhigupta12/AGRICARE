from django.core.paginator import Paginator
import fertilizer_prediction2
from django.shortcuts import redirect, render
from buysell.models import Form
import requests
import joblib
import tensorflow as tf


def cnn_fertilizer(request):
    tf.compat.v1.disable_eager_execution()
    xin = tf.compat.v1.placeholder('float', [None, 9])
    lis = []
    lis.append(float(request.POST.get('ca')))
    lis.append(float(request.POST.get('mg')))
    lis.append(float(request.POST.get('k')))
    lis.append(float(request.POST.get('s')))
    lis.append(float(request.POST.get('n')))
    lis.append(float(request.POST.get('lime')))
    lis.append(float(request.POST.get('c')))
    lis.append(float(request.POST.get('p')))
    lis.append(float(request.POST.get('moisture')))
    ab = fertilizer_prediction2.train_neural_network(xin, lis)
    s = " "
    return render(request, 'fertilizer_result.html', {'ans': s})


def post(request):
    name = request.POST.get('name')
    phone_no = request.POST.get('phone_no')
    city = request.POST.get('city')
    state = request.POST.get('state')
    crop_name = request.POST.get('crop_name')
    quantity = request.POST.get('quantity')
    price = request.POST.get('price')
    temp = Form(name=name, phone_no=phone_no, city=city,
                state=state, crop_name=crop_name, quantity=quantity, price=price)
    temp.save()
    # time.sleep(5)
    return redirect('home')


def display(request):
    obj = Form.objects.all()
    return render(request, 'sell.html', locals())


def gethome(request):
    return render(request, 'index.html')


def getlist(request):
    return render(request, 'list.html')


def get_recommendation(request):
    return render(request, 'recommend.html')


def get_prediction(request):
    return render(request, 'fprediction.html')


def get_crop_result(request):
    cls = joblib.load('final_model.sav')

    lis = []
    lis.append(request.POST.get('n'))
    lis.append(request.POST.get('p'))
    lis.append(request.POST.get('k'))
    lis.append(request.POST.get('temp'))
    lis.append(request.POST.get('hum'))
    lis.append(request.POST.get('ph'))
    lis.append(request.POST.get('rain'))

    ans = cls.predict([lis])
    s = ""
    s = s.join(ans)
    return render(request, 'crop_result.html', {'ans': s})


def get_fertilizer_result(request):
    cls = joblib.load('fertilizer_model.sav')

    lis = []
    lis.append(request.POST.get('ca'))
    lis.append(request.POST.get('mg'))
    lis.append(request.POST.get('k'))
    lis.append(request.POST.get('s'))
    lis.append(request.POST.get('n'))
    lis.append(request.POST.get('lime'))
    lis.append(request.POST.get('c'))
    lis.append(request.POST.get('p'))
    lis.append(request.POST.get('moisture'))

    ans = str(cls.predict([lis]))
    s = ""
    s = s.join(ans)
    return render(request, 'fertilizer_result.html', {'ans': s})


def live(request):
    main_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=json&offset="
    end_url = "&limit=10"
    data = []
    for offset in range(0, 270, 10):
        url = main_url + str(offset) + end_url
        response = requests.get(url).json()
        data += response["records"]

    p = Paginator(data, 10)
    page = request.GET.get('page')
    record = p.get_page(page)

    return render(request, 'mandi_live.html', {'response': record})
