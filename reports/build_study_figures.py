"""Build publication-style SVGs from the compact study evidence."""
from pathlib import Path
import csv
import importlib.util
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'reports/study'
DATA=STUDY/'data'
OUT=ROOT/'media/figures'
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('study_city',ROOT/'src/env/city.py')
city=importlib.util.module_from_spec(spec)
spec.loader.exec_module(city)

INK='#25344a'; MUTED='#526174'; GRID='#dfe5ec'; PAPER='#ffffff'
COLOR={'dsac_mlp':'#007f84','dsac_gru':'#264b90','ppo_mlp':'#bd632b','ppo_gru':'#8053a0'}
NAME={'dsac_mlp':'DSAC MLP','dsac_gru':'DSAC GRU','ppo_mlp':'PPO MLP','ppo_gru':'PPO GRU'}
MODELS=tuple(COLOR)
SID=1000002
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':INK,
    'axes.labelcolor':INK,'axes.edgecolor':GRID,'xtick.color':MUTED,'ytick.color':MUTED,
    'svg.fonttype':'none','svg.hashsalt':'gamblers-taxi-study','figure.facecolor':PAPER,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':PAPER})


def save(fig,name,description):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in fig.axes:
        for patch, labels in getattr(axis, '_study_boxes', []):
            rect = patch.get_extents()
            for label in labels:
                bounds = label.get_window_extent(renderer)
                if not (rect.x0 + 4 <= bounds.x0 and bounds.x1 <= rect.x1 - 4
                        and rect.y0 + 4 <= bounds.y0 and bounds.y1 <= rect.y1 - 4):
                    raise ValueError(f'{name}: text exceeds its box: {label.get_text()} ({bounds.bounds!r} vs {rect.bounds!r})')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.25,
                metadata={'Title':name.replace('_',' ').title(),'Description':description,'Date':None})
    if '--preview' in sys.argv:
        fig.savefig(OUT/(name+'.png'),dpi=105,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)


def rows(file):
    with np.load(file,allow_pickle=False) as data:
        ids=sorted({int(key.split('_',1)[0]) for key in data.files})
        return {sid:{key.split('_',1)[1]:data[key] for key in data.files
                     if key.startswith(f'{sid}_')} for sid in ids}


def base_map(ax):
    ax.set(xlim=(0,39),ylim=(0,29),aspect='equal',
           xticks=(0,10,20,30),yticks=(0,10,20))
    ax.tick_params(length=2,pad=2,labelsize=7)
    ax.grid(color='#eef1f5',linewidth=.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    for x,y in city.WALLS:
        ax.add_patch(Circle((x,y),1,facecolor='#d8dfe6',edgecolor='none',zorder=1))
    for (x0,x1),(_,y) in zip(city.TRAFFIC_LIMITS,city.TRAFFIC):
        ax.plot([x0,x1],[y,y],ls=(0,(3,3)),lw=.65,color='#aab4c0',zorder=1)
    names=('Home','Work','Bank','Garage','Slots','Blackjack','Fuel','Diner','Cafe','Restaurant')
    for (x,y),name in zip(city.ZONES,names):
        ax.scatter([x],[y],s=12,facecolor='white',edgecolor='#526174',lw=.7,zorder=5)
        align='right' if name in ('Cafe','Restaurant','Slots') else 'center'
        dx=-1.1 if name in ('Cafe','Restaurant','Slots') else 0.
        dy=-1.6 if name in ('Fuel','Diner','Slots','Blackjack') else 1.22
        ax.text(x+dx,y+dy,name,fontsize=6.7,ha=align,va='center',
                zorder=6,bbox={'facecolor':'white','edgecolor':'none','alpha':.79,'pad':.65})


def paths(ax,recordings,color,focus=None):
    for sid,record in recordings.items():
        p=record['position']
        if sid==focus:
            continue
        ax.plot(*p.T,color=color,alpha=.17,lw=.85,zorder=2)
        ax.scatter(*p[-1],s=8,color=color,alpha=.24,lw=0,zorder=3)
    if focus in recordings:
        record=recordings[focus];p=record['position']
        ax.plot(*p.T,color=color,lw=1.55,alpha=.92,zorder=4)
        glyph='o' if record['outcome'][-1]==1 else 'x'
        ax.scatter(*p[-1],marker=glyph,s=29,color=color,lw=1,zorder=5)


def match_maps():
    fig,axes=plt.subplots(2,2,figsize=(9,6.8),layout='constrained')
    fig.suptitle('Routes on 16 shared development scenarios',fontsize=14,fontweight='semibold')
    for ax,kind in zip(axes.flat,MODELS):
        collection=rows(DATA/f'{kind}_development_routes.npz')
        base_map(ax)
        paths(ax,collection,COLOR[kind],SID)
        wins=sum(row['outcome'][-1]==1 for row in collection.values())
        ax.set_title(f'{NAME[kind]} ({wins}/16 ownership)',loc='left',fontsize=10)
    fig.supxlabel('Map x (units)')
    fig.supylabel('Map y (units)')
    fig.text(.5,-.025,'Solid line: scenario 1000002     •     Open-area markers: service locations     '
             '•     Route end: circle for ownership, cross otherwise',ha='center',fontsize=8,color=MUTED)
    save(fig,'matched_paths','Active vehicle positions for the same first 16 scenario IDs. Solid line is scenario 1000002.')


def phase_maps():
    phases=[('dsac_gru_training_stage_00','Early training (ownership)','#527da9'),
            ('dsac_gru_training_stage_06','Training (food approach)','#007f84'),
            ('dsac_gru_training_stage_12','Training ($500 home start)','#51609b'),
            ('dsac_mlp_development','DSAC MLP (original)','#007f84'),
            ('dsac_mlp_forward_kl','DSAC MLP (forward-KL fork)','#8053a0'),
            ('dsac_gru_development','DSAC GRU (selected)','#264b90')]
    fig,axes=plt.subplots(2,3,figsize=(12,6.4),layout='constrained')
    for ax,(stem,title,color) in zip(axes.flat,phases):
        base_map(ax)
        collection=rows(DATA/(stem.replace('_development','_development_routes')+'.npz')
                        if stem!='dsac_mlp_forward_kl' else DATA/(stem+'_routes.npz'))
        paths(ax,collection,color,focus=SID if SID in collection else None)
        ax.set_title(title,loc='left',fontsize=9.5)
    fig.suptitle('Training fragments and frozen-policy routes',fontsize=13,fontweight='semibold')
    fig.text(.5,-.028,'Top: selected DSAC GRU training fragments   •   Bottom: matched development scenarios',
             ha='center',fontsize=8,color=MUTED)
    save(fig,'training_and_transfer_paths','Top panels show training fragments, not completed episodes. Bottom panels use matched scenario IDs.')


def resource_panel():
    fig,axes=plt.subplots(2,2,figsize=(8.8,5.1),layout='constrained')
    columns=(('satiety','Satiety'),('fuel','Fuel (units)'),
             ('cash','Cash ($)'),('car_debt','Car debt ($)'))
    for kind in MODELS:
        row=rows(DATA/f'{kind}_development_routes.npz')[SID]
        for ax,(field,label) in zip(axes.flat,columns):
            ax.plot(row['time'],row[field],lw=1.4,alpha=.83,color=COLOR[kind])
            ax.scatter([row['time'][-1]],[row[field][-1]],s=14,
                       marker='o' if row['outcome'][-1]==1 else 'x',color=COLOR[kind],lw=.9)
            ax.set_ylabel(label)
            ax.grid(color='#e9edf2',linewidth=.5)
    fig.supxlabel('Physical tick')
    fig.suptitle('Resources and debt on scenario 1000002',fontsize=13,fontweight='semibold')
    handles=[Line2D([],[],color=COLOR[k],label=NAME[k],lw=1.5) for k in MODELS]
    fig.legend(handles=handles,loc='outside lower center',ncols=4,frameon=False)
    save(fig,'resource_traces','Per-tick resources and car debt on an illustrative matched development scenario.')


def outcomes():
    with (DATA/'development_summary.csv').open(newline='',encoding='utf-8') as f:
        values={r['model']:r for r in csv.DictReader(f)}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.8,2.6),layout='constrained',sharey=True)
    y=np.arange(len(MODELS))
    labels=[NAME[k] for k in MODELS]
    for sub,field,maximum,title in (
            (ax,'ownership',64,'Ownership completions'),
            (bx,'lessons_passed',14,'Curriculum gates passed')):
        sub.barh(y,[maximum]*len(y),color='#eef1f5',height=.53)
        for i,k in enumerate(MODELS):
            n=int(values[k][field])
            sub.barh(i,n,color=COLOR[k],height=.53,alpha=.85)
            sub.text(min(n+maximum*.035,maximum*.89),i,f'{n}/{maximum}',va='center',fontsize=8)
        sub.set(xlim=(0,maximum*1.05),yticks=y,yticklabels=labels)
        sub.set_title(title,loc='left',fontsize=10)
        sub.grid(axis='x',color=GRID,linewidth=.55)
        sub.set_axisbelow(True)
    ax.invert_yaxis()
    save(fig,'development_results','Selected development ownership and training gates per architecture. Not final test performance.')


def curriculum_progress():
    fig,ax=plt.subplots(figsize=(8.8,3.8),layout='constrained')
    styles=('-', '-', '--', ':')
    for kind,style in zip(MODELS,styles):
        run=DATA/'cohort'/kind
        with (run/'training.jsonl').open(encoding='utf-8') as f:
            decisions={row['physical_transitions']:row['network_decisions_total']
                       for row in map(json.loads,f)}
        with (run/'curriculum_gates.jsonl').open(encoding='utf-8') as f:
            passed=[row for row in map(json.loads,f) if row['passed']]
        assert [row['stage'] for row in passed]==list(range(len(passed))),kind
        x=[0]+[decisions[row['physical_transitions']]/1000 for row in passed]
        end=max(decisions.values())/1000
        y=list(range(len(passed)+1))
        ax.step(x+[end],y+[y[-1]],where='post',color=COLOR[kind],
                linestyle=style,linewidth=1.6,label=NAME[kind])
        ax.scatter(end,y[-1],color=COLOR[kind],s=22,zorder=3)
    ax.set(xlim=(0,270),ylim=(0,14.7),yticks=range(0,15,2),
           xlabel='Network decisions (thousands)',ylabel='Curriculum gates passed')
    ax.set_title('Progress through the training curriculum',loc='left',fontsize=12,weight='semibold')
    ax.legend(loc='lower right',frameon=False,ncols=2)
    ax.grid(color=GRID,linewidth=.55)
    ax.set_axisbelow(True)
    save(fig,'curriculum_progress','Cumulative gates passed by recorded network decision count. Dots mark the ends of the four training runs.')


def final_condition_outcomes():
    with (DATA/'final_dsac_gru/summary.csv').open(newline='',encoding='utf-8') as f:
        summary=list(csv.DictReader(f))
    assert [row['condition'] for row in summary]==[f'S{i:02d}' for i in range(1,31)]
    fields=('ownership','starvation','collision','deadline','censored')
    colors=('#264b90','#c78348','#865499','#7e8d9f','#dce3eb')
    labels=('Ownership','Starvation','Collision','Deadline','Censored')
    assert all(int(row['episodes'])==256==sum(int(row[key]) for key in fields) for row in summary)
    fig,axes=plt.subplots(1,2,figsize=(12,6.1),layout='constrained')
    for ax,cases in zip(axes,(summary[:15],summary[15:])):
        y=np.arange(len(cases))
        counts=np.array([[int(row[key]) for key in fields] for row in cases])
        left=np.zeros(len(cases))
        names=[f"{row['condition']}  {row['case'].replace('_',' ').capitalize()}" for row in cases]
        for values,color in zip(counts.T,colors):
            ax.barh(y,values,left=left,color=color,height=.65)
            left+=values
        for index,n in enumerate(counts[:,0]):
            ax.text(263,index,str(n),va='center',fontsize=8)
        ax.text(263,1.005,'Owned',fontsize=8,transform=ax.get_xaxis_transform())
        ax.set(xlim=(0,289),xticks=(0,64,128,192,256),yticks=y,yticklabels=names,
               xlabel='Episodes',ylim=(len(cases)-.5,-.5))
        ax.tick_params(axis='y',length=0,labelsize=8)
        ax.grid(axis='x',color=GRID,linewidth=.55)
        ax.set_axisbelow(True)
    fig.suptitle('DSAC GRU outcomes across 30 frozen conditions',fontsize=13,weight='semibold')
    handles=[Line2D([],[],color=color,linewidth=6,label=label) for color,label in zip(colors,labels)]
    fig.legend(handles=handles,loc='outside lower center',ncols=5,frameon=False)
    save(fig,'final_condition_outcomes','Terminal outcome counts in each of 30 registered conditions, with 256 episodes per condition.')


def box(ax,x,y,w,h,title,body,edge='#b5c3d4',fill='#f7f9fc'):
    patch = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.018,rounding_size=.08',
                           fc=fill,ec=edge,lw=.9)
    ax.add_patch(patch)
    heading = ax.text(x+.15,y+h-.18,title,fontsize=10,fontweight='semibold',va='top')
    details = ax.text(x+.15,y+h-.50,body,fontsize=8,va='top',linespacing=1.35,color=MUTED)
    if not hasattr(ax, '_study_boxes'):
        ax._study_boxes = []
    ax._study_boxes.append((patch, (heading, details)))


def arrow(ax,start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,
                                fc='#6d7f94',ec='#6d7f94',lw=.85))


def pipeline():
    fig,ax=plt.subplots(figsize=(10.8,5.4))
    ax.set(xlim=(0,11),ylim=(0,5.4));ax.axis('off')
    ax.text(.2,5.24,'Training and evaluation',fontsize=15,weight='semibold')
    box(ax,.2,3.73,2.25,1.03,'JAX environment','Vehicle and city state\nReward and outcome')
    box(ax,2.97,3.73,2.3,1.03,'Observation','Lidar and distances\nBalances and clock')
    box(ax,5.8,3.73,2.05,1.03,'Hybrid actor','Categorical intention\nGaussian driving style')
    box(ax,8.38,3.73,2.42,1.03,'Route controller','Steering and acceleration\nUp to 128 ticks per decision')
    for x,y in ((2.45,2.97),(5.27,5.8),(7.85,8.38)):arrow(ax,(x,4.23),(y,4.23))
    arrow(ax,(9.56,3.73),(9.56,3.31))
    box(ax,8.38,2.22,2.42,1.08,'Experience','64 lanes × 8 decisions\nDiscounted by duration')
    arrow(ax,(8.38,2.74),(7.85,2.74))
    box(ax,5.8,2.22,2.05,1.08,'Learner','DSAC: quantile replay\nPPO: rollout and GAE')
    arrow(ax,(6.8,3.3),(6.8,3.73))
    arrow(ax,(5.8,2.74),(5.27,2.74))
    box(ax,2.97,2.22,2.3,1.08,'Development gate','Lesson and skill checks\nSave evaluated weights')
    arrow(ax,(2.97,2.74),(2.45,2.74))
    box(ax,.2,2.22,2.25,1.08,'Frozen evaluation','Fixed policies and seeds\nAfter cohort readiness')
    box(ax,.2,.57,4.05,1.02,'Curriculum resets','Change location, debt and starting resources.')
    box(ax,4.68,.57,6.12,1.02,'Saved outputs','Episode logs supply results, figures and video.\nCheckpoints store settings and source hash.')
    save(fig,'training_pipeline','JAX simulator and hybrid learning with separate development readiness and final evaluation.')


def curriculum():
    groups=[('Settle debt','Garage and bank'),('Earning','Work and payment'),
            ('Food','Purchase and travel'),('Resources','Fuel, fatigue,\npersonal debt'),
            ('Home starts','$250, $350, $500'),('Full campaign','Start at home')]
    fig,ax=plt.subplots(figsize=(11.8,2.8))
    ax.set(xlim=(0,12),ylim=(0,2.7));ax.axis('off')
    ax.text(.1,2.43,'Training curriculum',fontsize=13,weight='semibold')
    for i,(label,body) in enumerate(groups):
        x=.1+i*1.98
        box(ax,x,.88,1.77,1.02,label,body,fill='#f2f6fa')
        if i<len(groups)-1:arrow(ax,(x+1.77,1.37),(x+1.98,1.37))
    ax.text(.1,.44,'The four runs used the same 14-lesson schedule and a limit of 262,144 decisions each.',
            fontsize=8.6,color=MUTED)
    save(fig,'ownership_curriculum','Six groups of the historical 14-stage reset curriculum.')


def marker(ax, x, y, width):
    ax.plot([x, x + width*.31, x + width*.67, x + width],
            [y, y+.023, y-.014, y+.009], color='#d5a637', lw=5,
            alpha=.40, solid_capstyle='round', zorder=2)


def architectures():
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.1))
    fig.subplots_adjust(left=.045,right=.98,top=.94,bottom=.045,hspace=.34)
    for ax, dsac in zip(axes, (True, False)):
        ax.set(xlim=(0,12),ylim=(0,3.65))
        ax.axis('off')
        name = 'DSAC' if dsac else 'PPO'
        ax.text(.1,3.57,f'{name} MLP and {name} GRU',fontsize=13,
                weight='semibold',va='top')
        box(ax,.1,1.89,2.02,1.15,'Observation','194 public features\nLegal-action mask')
        memory = 'GRU: 32-decision window' if dsac else 'GRU: state across decisions'
        box(ax,2.50,1.89,2.72,1.15,'State encoder',f'MLP: current input\n{memory}')
        marker(ax,2.64,2.20,1.67 if dsac else 1.82)
        box(ax,5.61,1.89,2.88,1.15,'Action heads','Categorical service choice\nGaussian driving style')
        box(ax,8.88,1.89,3.00,1.15,'Collect experience','Controller drives the car\nSave reward and next state')
        for left,right in ((2.12,2.5),(5.22,5.61),(8.49,8.88)):
            arrow(ax,(left,2.47),(right,2.47))

        actor = 'Actor update','Use critics and entropy\nSeparate control temperatures'
        critic = 'Critic update','Fit twin quantile critics\nUpdate the target networks'
        data = 'Replay buffer','Sample stored transitions\nInclude history for GRU'
        if not dsac:
            actor = 'Joint PPO update','Clipped policy objective\nJoint policy and value update'
            critic = 'Rollout advantages','Use the saved value estimates\nCompute GAE and returns'
            data = 'Rollout batch','Use the latest policy rollout\nInclude recurrent state for GRU'
        for x,w,(title,body) in ((8.88,3.0,data),(5.61,2.88,critic),(2.50,2.72,actor)):
            box(ax,x,.18,w,1.13,title,body,fill='#edf3f8')
        marker(ax,9.02,.64,1.61 if dsac else 1.92)
        arrow(ax,(10.38,1.89),(10.38,1.32))
        arrow(ax,(8.88,.75),(8.50,.75))
        arrow(ax,(5.61,.75),(5.23,.75))
        arrow(ax,(3.86,1.31),(3.86,1.88))
    save(fig,'agent_architectures','Model inputs, action heads and optimizer loops for DSAC and PPO. Marker strokes identify recurrent state and the source of update data.')


def environment():
    fig,ax=plt.subplots(figsize=(10.4,3.8))
    ax.set(xlim=(0,11),ylim=(0,3.8));ax.axis('off')
    ax.text(.2,3.55,'Environment and campaign rules',fontsize=14,weight='semibold')
    box(ax,.2,1.85,3.15,1.17,'Vehicle dynamics','Acceleration and steering\nFriction, fatigue and collisions')
    box(ax,3.89,1.85,3.15,1.17,'Services','Timed work, meals, fuel and sleep\nBank payments and casino games')
    box(ax,7.58,1.85,3.15,1.17,'Campaign outcome','Garage after both debts are clear\nCollision, hunger or deadline')
    arrow(ax,(3.35,2.43),(3.89,2.43))
    arrow(ax,(7.04,2.43),(7.58,2.43))
    ax.text(.2,1.2,'Every physical tick updates time, resource use, traffic and liabilities.',fontsize=9,color=MUTED)
    ax.text(.2,.72,'Observation: lidar, service bearings, resources, accounts, public prices and clock.',
            fontsize=9,color=MUTED)
    save(fig,'environment_model','Relationship between kinematic motion, service actions and campaign termination.')


if __name__=='__main__':
    for render in (match_maps,phase_maps,resource_panel,outcomes,pipeline,curriculum,architectures,environment,
                   curriculum_progress,final_condition_outcomes):
        render()
    print('Generated',len(list(OUT.glob('*.svg'))),'SVG figures.')
