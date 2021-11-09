import argparse
def eval(lr_metrics, xgb_metrics):
    import joblib
    import pandas as pd
    import pprint
    
    lr_metrics=joblib.load(lr_metrics)
    xgb_metrics=joblib.load(xgb_metrics)

    lis = [lr_metrics, xgb_metrics]
    max = 0
    for i in lis:
        accuracy = i['score']
        if accuracy > max:
            max = accuracy
            model = i['model']
            metrics = i
  
    print('Best metrics \n\n')
    pprint.pprint(metrics)

    joblib.dump(model, 'best_model')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_metrics')
    parser.add_argument('--xgb_metrics')
    args = parser.parse_args()
    eval(args.lr_metrics,args.xgb_metrics)
