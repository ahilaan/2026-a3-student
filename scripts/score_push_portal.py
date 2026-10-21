"""Authorized portal submission/status helper; no credentials in source or ZIP."""
import argparse
import datetime as dt
import email.utils
import hashlib
import http.cookiejar
import json
from pathlib import Path
import urllib.request
import urllib.error
import uuid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/compliant_campaign'
BASE='http://asterix.cse.iitd.ac.in:8123'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['status','upload','result','choose'])
    ap.add_argument('--id',type=int);ap.add_argument('--sha256')
    ap.add_argument('--compact',action='store_true');args=ap.parse_args()
    jar=http.cookiejar.MozillaCookieJar(str(ROOT/'tmp/course_cookies.txt'))
    jar.load(ignore_discard=True,ignore_expires=True)
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    server_now=None
    def get(path):
        nonlocal server_now
        with opener.open(BASE+path,timeout=60) as response:
            server_now=email.utils.parsedate_to_datetime(response.headers['Date'])
            return json.load(response)
    def save(name,value):
        (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    session=get('/api/session')
    assert session['username'].lower()=='2023cs10076' and session['team_name']=='AHILAAN_SAXENA'
    if args.mode=='status':
        board=get('/api/leaderboard');history=get('/api/my-submissions')
        save('leaderboard_latest.json',board);save('history_latest.json',history)
        own=next(r for r in board['rows'] if r['team']=='AHILAAN_SAXENA')
        print(json.dumps({'own':own,'leader':board['rows'][0],'server_utc':server_now.isoformat()},indent=2))
    elif args.mode=='result':
        assert args.id
        result=get('/api/submissions/'+str(args.id));save('submission_'+str(args.id)+'.json',result)
        displayed = ({k:result.get(k) for k in ('id','status','message','ltr_dev_ndcg','c2_mean',
                     'checks_passed','checks_total','counts_on_leaderboard')} if args.compact else result)
        print(json.dumps(displayed,indent=2))
    elif args.mode=='choose':
        if not args.id:
            ap.error('choose requires --id')
        history=get('/api/my-submissions')
        assert history.get('can_choose'),'Portal selection is closed'
        selected=next(r for r in history['rows'] if r['id']==args.id)
        assert selected['status']=='success' and selected['files_ok']
        request=urllib.request.Request(BASE+f'/api/submissions/{args.id}/final',data=b'')
        with opener.open(request,timeout=60) as response:receipt=json.load(response)
        save('choose_'+str(args.id)+'.json',receipt)
        print(json.dumps(receipt,indent=2))
    else:
        config=get('/api/config');assert config['submissions_open'],config.get('closed_reason')
        history=get('/api/my-submissions')
        pending=[r['id'] for r in history['rows'] if r['status'] in ('queued','running')]
        if pending:
            raise SystemExit('Wait for active portal submissions before uploading: '+str(pending))
        if history['rows']:
            last=max(dt.datetime.fromisoformat(r['submitted_at']) for r in history['rows'])
            assert (server_now-last).total_seconds()>=60*config['cooldown_minutes']+3,'Portal cooldown has not elapsed'
        data=(ROOT/'2023CS10076.zip').read_bytes();digest=hashlib.sha256(data).hexdigest()
        assert args.sha256==digest and len(data)<=config['max_upload_mb']*1024**2
        boundary=uuid.uuid4().hex
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="2023CS10076.zip"\r\nContent-Type: application/zip\r\n\r\n'.encode()+data+f'\r\n--{boundary}--\r\n'.encode())
        request=urllib.request.Request(BASE+'/api/submissions',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
        try:
            with opener.open(request,timeout=120) as response:receipt=json.load(response)
        except urllib.error.HTTPError as exc:
            detail=exc.read().decode('utf-8',errors='replace')
            raise SystemExit(f'Portal rejected upload (HTTP {exc.code}): {detail}') from None
        save('upload_'+str(receipt['id'])+'.json',{'receipt':receipt,'sha256':digest,'bytes':len(data),
                                               'pending_before_upload':pending})
        print(json.dumps(receipt,indent=2))
        return receipt


if __name__=='__main__':main()
