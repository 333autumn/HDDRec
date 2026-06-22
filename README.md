## Enviroment Requirement

`pip install -r requirements.txt`

## Dataset

We provide three processed datasets and the corresponding knowledge graphs: Yelp2018 and Amazon-book and MIND.

## Model Variants

We also simply implement LightGCN (*SIGIR'20*) and SGL (*SIGIR'21*) for easy comparison. You can test these models implemented here by:

` cd code && python main.py --dataset=yelp2018 --model=lgn `

and

` cd code && python main.py --dataset=yelp2018 --model=sgl `

However, we still recommend to also refer to the authors' official implementation to avoid potential performance problems.
