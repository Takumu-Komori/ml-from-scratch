### 実行


## 初回
# pytorchのインストール(CPU only)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# 他のライブラリをインストール
pip install -r requirements.txt
# インストール確認
python - <<EOF
import torch
print(torch.__version__)
print(torch.randn(2,3))
EOF
# script実行
python -m src.linear_regression \
  --n 512 \
  --d 3 \
  --noise_std 0.1 \
  --lr 0.1 \
  --steps 300 \
  --seed 42





# ml-from-scratch
python -m src.linear_regression --steps 300 --lr 0.1 --d 3

set_seed関数<br>
python、numpy、pytorchでの乱数を固定する。これにより、結果の再現性を担保する。また、初期値を固定することで意図的に変更した際にそれ以外の条件の同一に保ち、挙動変化を意図的な変化の結果と解釈しやすくなる。

make_symnthetic_regression関数<br>
make_synthetic_regression は、真の重み・バイアスが既知な線形モデルに基づき、観測ノイズを含む回帰データを生成する関数である。これにより、学習によって推定されたパラメータが真値にどの程度近づくかを定量的に検証できる。

gは今関数における乱数の使用の際、seedを固定したlocalな乱数を使用させることで再現性を保証する
xは教師入力値
true_w, true_bは真の重みと真のバイアス
yは真の推論モデル（回帰関数）＋観察ノイズ
後述の回帰モデルの目的はノイズを含まない真の回帰関数を最小二乗の意味での近似をすることである

LinearRegressionクラス<br>
_init_関数<br>
学習される変数wとbをtorch.nn.Parameterとして宣言し、optimizerが更新対象として認識できる様にする
super.__init__()はpytorchのmoduleの初期化によってParameter管理構造を生成する、これによりnn.Parameterはwとbを学習対象としてパラメーター登録することができる。
torchi.nn.parameterはwとbを学習対象のパラメーターとして登録するとともに今回は全て0の初期値を与える。
この過程を通してoptimizerが認識し更新可能な変数wとbを定義する

forward関数<br>
forward 関数は、パラメータ w,b による仮説関数 f(x)=xw+b を定義しており、最適化ではこの関数族の中から損失を最小化するものを探索している。


main()<br>

args<br>
	• --n：サンプル数 N <br>
	•	--d：特徴量次元 D <br>
	•	--noise_std：観測ノイズの標準偏差 \sigma <br>
	•	--lr：SGDの学習率 <br>
	•	--steps：更新回数 <br>
	•	--seed：再現性固定用 <br>
	•	--log_path：lossログ保存先 <br>
  
p＝argparse.ArgparseParserについて<br>
argparse によりハイパーパラメータをコマンドラインから指定することで、コードを変更せずに実験条件を切り替え、再現性のある比較実験を可能にしている。
コード改変が伴うとgit diffの汚染される(git上でコード変更があったのか条件変更があったのか区別できない)。コード改変は純粋にアルゴリズムや実装の変更だけの時に行われるべき
また、これにより実行コマンドラインとその出力の形で比較

