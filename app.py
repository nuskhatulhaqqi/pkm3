from flask import Flask, render_template, request, redirect,escape
import numpy as np                        # numpy==1.19.3
from flask import Response
import pandas as pd
import math as m

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    nama = "natul"
    display=0
    test = pd.read_csv("dataset/u1.test.csv")
    data_test = test.pivot(index = 'user_id', columns = 'item_id', values = 'rating').fillna(0)
    id=data_test.index
    return render_template('index.html',title='Home', nama=nama,id=id,display=display)

@app.route('/tidak_setara')
def tidak_setara():
    test = pd.read_csv("dataset/u1.test.csv")
    data_test = test.pivot(index = 'user_id', columns = 'item_id', values = 'rating').fillna(0)
    id=data_test.index
    return render_template('tidak_setara.html',title='Pembobotan Tidak Setara', id=id)

def data_tidak_setara():
  test = pd.read_csv("dataset/u1.test.csv")
  data_test = test.pivot(index = 'user_id', columns = 'item_id', values = 'rating').fillna(0)
  hasil_prediksi = pd.read_csv('dataset/f1/Predict_F1K5_SDUB.csv')
  data_training = pd.read_csv('dataset/data_training_baru.csv')
  gt = pd.read_csv("dataset/data_testing_t.csv")
  return hasil_prediksi,data_training,gt,data_test

def argmax_non_zero(similar):
    data = similar.to_numpy()
    result = []
    for row in data:
        # untuk mengambil index dari rating yang tidak 0 
        non_zero_indices = np.where(row != 0)[0]
        # untuk mengurutkan nilai rating dari terbesar ke kecil
        sorted_indices = np.argsort(-row[non_zero_indices])
        # untuk mengambil nilai indexnya setelah terurut 
        sorted_non_zero_indices = non_zero_indices[sorted_indices]
        # memasukkan hasil setiap baris dalam matrix
        result.append(sorted_non_zero_indices)
    return result

def ideal_dcg (N):
  idcg = 0
  log = []
  id = []
  hasil = []
  for n in range(1, N+1):
    idcg += 1/(m.log2(n + 1))
    log.append(1/(m.log2(n + 1)))
    id.append(idcg)
  hasil.append(log)
  hasil.append(id)
  return hasil

def normalized_dcg1 (target, data_test, topN, N):
  idcg_log = ideal_dcg (N)
  ndcg = []
  dcg = 0
  for n in range(1, N+1):
    item_test = data_test[target-1]
    if topN[target-1][n-1] in item_test:
      indicator = 1
    else:
      indicator = 0
    log = (idcg_log[0][n-1]) * indicator
    dcg += log
    dcg_akhir = (1/1)*dcg
    hasil=dcg_akhir/idcg_log[1][n-1]
    if hasil == 0:
      hasil = 0
    else:
      hasil = round(hasil, 3)
    ndcg.append(hasil)
  return ndcg

def get_recomendation(user_id, movie, top, N):
  film = []
  id = []
  for n in range(N):
    indeks_item = top[user_id][n]
    film.append(str(n+1)+" "+str(movie.iloc[indeks_item][1]))
    id.append(indeks_item)
  return film,id

def get_recomendation1(movie, top):
  film = []
  id = []
  for n in range(len(top)):
    indeks_item = top[n]
    film.append(str(n+1)+" "+str(movie.iloc[indeks_item][1]))
    id.append(indeks_item)
  return film,id
def get_recomendation2(user_id, movie, top, N):
  film = []
  id = []
  for n in range(N):
    indeks_item = top[user_id][n]
    film.append(str(n+1)+" "+str(movie.iloc[indeks_item][1]))
    id.append(indeks_item)
  return film,id

movie = pd.read_csv("dataset/u.item.csv", encoding='latin-1')

def irisan(test,rekomendasi):
    irisan = []
    for u in range(len(rekomendasi)):
        if (rekomendasi[u] in test):
            irisan.append(rekomendasi[u])
    return irisan

@app.route('/form_tidak_setara', methods=['POST','GET'])
def form_tidak_setara():
    if request.method == 'POST':
        hasil_prediksi,data_training,gt,data_test=data_tidak_setara()
        id=data_test.index
        #input data web
        target_user = int(request.form['target_user'])
        jumlah_n = int(request.form['jumlah_n'])
        #
        topN = argmax_non_zero(hasil_prediksi)
        ind_item = argmax_non_zero(gt)
        training_item = argmax_non_zero(data_training)
        ndcg1 = normalized_dcg1 (target_user, ind_item, topN, 100)
        ndcg=ndcg1[jumlah_n-1]
        #data training
        target_u = target_user-1
        dt,id_dt= get_recomendation2(target_u, movie, training_item, len(training_item[target_u]))
        #data test/ground trud
        dgt,id_dgt=get_recomendation2(target_u, movie, ind_item,len(ind_item[target_u]))
        # data rekomendasi
        dr,id_dr=get_recomendation(target_u, movie, topN, jumlah_n)
        # irisan 
        ir = irisan(id_dgt,id_dr)
        irr,id_irr=get_recomendation1(movie, ir)
        display = 1
        return render_template('tidak_setara.html',display=display,target_user=target_user,jumlah_n=jumlah_n,hasil_ndcg=ndcg,dr=dr,id_dr=id_dr,dt=dt,dgt=dgt,id_dgt=id_dgt,ir=irr,id_irr=id_irr,jir=len(irr),jdr=len(dr),jdt=len(dt),jdgt=len(dgt),id=id)
    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)