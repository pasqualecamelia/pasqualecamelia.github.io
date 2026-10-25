// QGT Hydrogen3D r06 — guided physical baseline
// Dimensionless Schrödinger-grade coordinates: rho=r/a_Q, tau=E_Q t/hbar
// Baseline: i dpsi/dtau = [-1/2 nabla^2 - 1/rho - F(tau) rho_z] psi
// Modes:
// PREPARE: imaginary-time eigenstate projection (numerical preparation; NOT physical radiative capture)
// LIVE: real-time unitary evolution of a 1s+2pz+3dz2 bound superposition
// LIGHT: real-time split-operator with an explicit near-resonant dipole pulse -F(tau) rho_z
(function(global){
'use strict';
var CFG={N:64,dx:0.375,dt:0.18,dtau:0.05,E0:0.12,omega:0.375,Tpulse:30.0};
function makeFFT(N){
  var levels=Math.log2(N)|0,rev=new Uint32Array(N),i;
  for(i=0;i<N;i++){var x=i,r=0;for(var b=0;b<levels;b++){r=(r<<1)|(x&1);x>>=1;}rev[i]=r;}
  var cosT=new Float64Array(N/2),sinT=new Float64Array(N/2);
  for(i=0;i<N/2;i++){cosT[i]=Math.cos(2*Math.PI*i/N);sinT[i]=Math.sin(2*Math.PI*i/N);}
  return function fft(re,im,inverse){var n=N,j,k,len,half,step;
    for(i=0;i<n;i++){j=rev[i];if(j>i){var tr=re[i];re[i]=re[j];re[j]=tr;var ti=im[i];im[i]=im[j];im[j]=ti;}}
    for(len=2;len<=n;len<<=1){half=len>>1;step=n/len;for(i=0;i<n;i+=len){for(j=i,k=0;j<i+half;j++,k+=step){var c=cosT[k],s=inverse?sinT[k]:-sinT[k],l=j+half;var xr=re[l]*c-im[l]*s,xi=re[l]*s+im[l]*c;re[l]=re[j]-xr;im[l]=im[j]-xi;re[j]+=xr;im[j]+=xi;}}}
    if(inverse)for(i=0;i<n;i++){re[i]/=n;im[i]/=n;}
  };
}
function Engine(){
  var N=CFG.N,N2=N*N,N3=N*N*N,dx=CFG.dx;
  this.N=N;this.cfg=CFG;this.mode='prepare';this.re=new Float64Array(N3);this.im=new Float64Array(N3);this.t=0;this.steps=0;
  this.xs=new Float64Array(N);for(var i=0;i<N;i++)this.xs[i]=(i-N/2+0.5)*dx;
  this.V=new Float64Array(N3);this.Zcoord=new Float64Array(N3);this.Rcoord=new Float64Array(N3);
  this.b1=new Float64Array(N3);this.b2=new Float64Array(N3);this.b3=new Float64Array(N3);
  var idx=0,a,b,c,eps=1e-12;
  for(a=0;a<N;a++)for(b=0;b<N;b++)for(c=0;c<N;c++,idx++){
    var xx=this.xs[a],yy=this.xs[b],zz=this.xs[c],r=Math.sqrt(xx*xx+yy*yy+zz*zz)+eps;
    this.Rcoord[idx]=r;this.Zcoord[idx]=zz;this.V[idx]=-1/r;
    this.b1[idx]=Math.exp(-r);
    this.b2[idx]=zz*Math.exp(-r/2);
    this.b3[idx]=(3*zz*zz-r*r)*Math.exp(-r/3);
  }
  this._normalizeReal(this.b1);this._normalizeReal(this.b2);this._normalizeReal(this.b3);
  var ks=new Float64Array(N);for(i=0;i<N;i++){var f=(i<=N/2?i:i-N)*(2*Math.PI/(N*dx));ks[i]=f;}
  this.krCos=new Float64Array(N3);this.krSin=new Float64Array(N3);this.ki=new Float64Array(N3);
  idx=0;for(a=0;a<N;a++)for(b=0;b<N;b++)for(c=0;c<N;c++,idx++){
    var k2=(ks[a]*ks[a]+ks[b]*ks[b]+ks[c]*ks[c])/2,ph=-k2*CFG.dt;
    this.krCos[idx]=Math.cos(ph);this.krSin[idx]=Math.sin(ph);this.ki[idx]=Math.exp(-k2*CFG.dtau);
  }
  this.fft=makeFFT(N);this.lre=new Float64Array(N);this.lim=new Float64Array(N);
  this.setMode('prepare');
}
Engine.prototype._normalizeReal=function(arr){var n=0;for(var i=0;i<arr.length;i++)n+=arr[i]*arr[i];n=Math.sqrt(n);for(i=0;i<arr.length;i++)arr[i]/=n;};
Engine.prototype._normalize=function(){var n=0;for(var i=0;i<this.re.length;i++)n+=this.re[i]*this.re[i]+this.im[i]*this.im[i];n=Math.sqrt(n);for(i=0;i<this.re.length;i++){this.re[i]/=n;this.im[i]/=n;}};
Engine.prototype.setMode=function(mode){this.mode=mode;this.t=0;this.steps=0;var i,N3=this.re.length;
  if(mode==='prepare'){
    var sig=3;for(i=0;i<N3;i++){var r=this.Rcoord[i];this.re[i]=Math.exp(-r*r/(2*sig*sig));this.im[i]=0;}this._normalize();
  } else if(mode==='life'){
    var s=1/Math.sqrt(3);for(i=0;i<N3;i++){this.re[i]=s*(this.b1[i]+this.b2[i]+this.b3[i]);this.im[i]=0;}this._normalize();
  } else {
    for(i=0;i<N3;i++){this.re[i]=this.b1[i];this.im[i]=0;}this._normalize();
  }
};
Engine.prototype._fftAxis=function(axis,inverse){var N=this.N,N2=N*N,re=this.re,im=this.im,lre=this.lre,lim=this.lim,fft=this.fft,a,b,i,base,stride;
  for(a=0;a<N;a++)for(b=0;b<N;b++){
    if(axis===0){base=a*N+b;stride=N2;}else if(axis===1){base=a*N2+b;stride=N;}else{base=a*N2+b*N;stride=1;}
    for(i=0;i<N;i++){lre[i]=re[base+i*stride];lim[i]=im[base+i*stride];}fft(lre,lim,inverse);for(i=0;i<N;i++){re[base+i*stride]=lre[i];im[base+i*stride]=lim[i];}
  }
};
Engine.prototype.lightField=function(){if(this.mode!=='light'||this.t<0||this.t>CFG.Tpulse)return 0;var env=Math.sin(Math.PI*this.t/CFG.Tpulse);env*=env;return CFG.E0*env*Math.cos(CFG.omega*this.t);};
Engine.prototype._realPotentialKick=function(halfdt){var F=this.lightField(),re=this.re,im=this.im,V=this.V,Z=this.Zcoord;for(var i=0;i<re.length;i++){var ph=-(V[i]-F*Z[i])*halfdt,c=Math.cos(ph),s=Math.sin(ph),r=re[i],q=im[i];re[i]=r*c-q*s;im[i]=r*s+q*c;}};
Engine.prototype._realStep=function(){var half=CFG.dt/2;this._realPotentialKick(half);this._fftAxis(0,false);this._fftAxis(1,false);this._fftAxis(2,false);for(var i=0;i<this.re.length;i++){var c=this.krCos[i],s=this.krSin[i],r=this.re[i],q=this.im[i];this.re[i]=r*c-q*s;this.im[i]=r*s+q*c;}this._fftAxis(0,true);this._fftAxis(1,true);this._fftAxis(2,true);this.t+=CFG.dt;this._realPotentialKick(half);this.steps++;};
Engine.prototype._imagStep=function(){var h=CFG.dtau/2,re=this.re,im=this.im,V=this.V;for(var i=0;i<re.length;i++){var f=Math.exp(-V[i]*h);re[i]*=f;im[i]*=f;}this._fftAxis(0,false);this._fftAxis(1,false);this._fftAxis(2,false);for(i=0;i<re.length;i++){re[i]*=this.ki[i];im[i]*=this.ki[i];}this._fftAxis(0,true);this._fftAxis(1,true);this._fftAxis(2,true);for(i=0;i<re.length;i++){f=Math.exp(-V[i]*h);re[i]*=f;im[i]*=f;}this._normalize();this.t+=CFG.dtau;this.steps++;};
Engine.prototype.step=function(){if(this.mode==='prepare'){for(var j=0;j<3;j++)this._imagStep();}else this._realStep();};
Engine.prototype.norm=function(){var n=0;for(var i=0;i<this.re.length;i++)n+=this.re[i]*this.re[i]+this.im[i]*this.im[i];return n;};
Engine.prototype._overlapReal=function(basis){var rr=0,ii=0;for(var i=0;i<this.re.length;i++){rr+=basis[i]*this.re[i];ii+=basis[i]*this.im[i];}return rr*rr+ii*ii;};
Engine.prototype.observables=function(){var den=0,mr=0,z=0;for(var i=0;i<this.re.length;i++){var p=this.re[i]*this.re[i]+this.im[i]*this.im[i];den+=p;mr+=p*this.Rcoord[i];z+=p*this.Zcoord[i];}return {norm:den,meanR:mr/den,z:z/den,p1:this._overlapReal(this.b1)/den,p2:this._overlapReal(this.b2)/den,field:this.lightField()};};
Engine.prototype.voxels=function(maxCount){var N=this.N,N2=N*N,N3=N*N*N,re=this.re,im=this.im,xs=this.xs,dx=CFG.dx;if(!this._pArr)this._pArr=new Float32Array(N3);var pA=this._pArr,pmax=0,i;for(i=0;i<N3;i++){var q=re[i]*re[i]+im[i]*im[i];pA[i]=q;if(q>pmax)pmax=q;}var thr=pmax*0.0012,count,pass;for(pass=0;pass<7;pass++){count=0;for(i=0;i<N3;i++)if(pA[i]>thr)count++;if(count<=maxCount)break;thr*=1.8;}var X=new Float32Array(count),Y=new Float32Array(count),Z=new Float32Array(count),P=new Float32Array(count),PH=new Float32Array(count),JX=new Float32Array(count),JY=new Float32Array(count),JZ=new Float32Array(count);var n=0,jmax=0,JIT=.35*dx,RCLIP=10.5;
  for(var a=1;a<N-1;a++)for(var b=1;b<N-1;b++)for(var c=1;c<N-1;c++){if(Math.abs(xs[a])>RCLIP||Math.abs(xs[b])>RCLIP||Math.abs(xs[c])>RCLIP)continue;var idx=a*N2+b*N+c;if(pA[idx]>thr&&n<count){X[n]=xs[a]+(Math.random()-.5)*2*JIT;Y[n]=xs[b]+(Math.random()-.5)*2*JIT;Z[n]=xs[c]+(Math.random()-.5)*2*JIT;P[n]=pA[idx]/pmax;PH[n]=Math.atan2(im[idx],re[idx]);var jx=re[idx]*(im[idx+N2]-im[idx-N2])-im[idx]*(re[idx+N2]-re[idx-N2]);var jy=re[idx]*(im[idx+N]-im[idx-N])-im[idx]*(re[idx+N]-re[idx-N]);var jz=re[idx]*(im[idx+1]-im[idx-1])-im[idx]*(re[idx+1]-re[idx-1]);JX[n]=jx;JY[n]=jy;JZ[n]=jz;var jm=Math.sqrt(jx*jx+jy*jy+jz*jz);if(jm>jmax)jmax=jm;n++;}}
  return {n:n,x:X,y:Y,z:Z,p:P,ph:PH,jx:JX,jy:JY,jz:JZ,jmax:jmax};
};
Engine.prototype.slices=function(){var N=this.N,N2=N*N,h=N>>1,re=this.re,im=this.im;function blank(){var m=[];for(var i=0;i<N;i++)m.push(new Float64Array(N));return m;}var xy=blank(),xz=blank(),yz=blank(),max=0,a,b;for(a=0;a<N;a++)for(b=0;b<N;b++){var i1=a*N2+b*N+h,i2=a*N2+h*N+b,i3=h*N2+a*N+b;xy[a][b]=re[i1]*re[i1]+im[i1]*im[i1];xz[a][b]=re[i2]*re[i2]+im[i2]*im[i2];yz[a][b]=re[i3]*re[i3]+im[i3]*im[i3];max=Math.max(max,xy[a][b],xz[a][b],yz[a][b]);}var sc=max>0?255/max:0;for(a=0;a<N;a++)for(b=0;b<N;b++){xy[a][b]=Math.round(xy[a][b]*sc);xz[a][b]=Math.round(xz[a][b]*sc);yz[a][b]=Math.round(yz[a][b]*sc);}return {xy:xy,xz:xz,yz:yz};};
global.QGTHydrogenR06=Engine;
})(typeof window!=='undefined'?window:globalThis);

// ---------- UI / renderer ----------
var r06Eng=null,r06Frames=[],r06Fi=0,r06Playing=false,r06Timer=null,r06Mode='prepare',r06Hist=[];
var R06_LIMITS={prepare:40,life:100,light:180};
function r06Snapshot(){var s=r06Eng.slices();return {t:r06Eng.t,obs:r06Eng.observables(),xy:s.xy,xz:s.xz,yz:s.yz,vox:r06Eng.voxels(18000)};}
var r06Rot=.25,r06Tilt=1.15,r06SpritesCache=null;
function r06Sprites(){if(r06SpritesCache)return r06SpritesCache;function mk(h,w){var c=document.createElement('canvas');c.width=c.height=32;var g=c.getContext('2d'),col=w?'rgba(255,255,255,':'hsla('+h+',95%,62%,';var gr=g.createRadialGradient(16,16,0,16,16,16);gr.addColorStop(0,col+'1)');gr.addColorStop(.35,col+'.55)');gr.addColorStop(1,col+'0)');g.fillStyle=gr;g.fillRect(0,0,32,32);return c;}var arr=[],line=[];for(var i=0;i<12;i++){arr.push(mk(200+i*30,false));line.push('hsl('+(200+i*30)+',95%,68%)');}return r06SpritesCache={hue:arr,white:mk(0,true),line:line};}
function r06Band(p){var w=0,B=[[.45,.07,.7],[.13,.03,1],[.035,.010,1.4],[.008,.0028,1.1]];for(var i=0;i<4;i++){var d=(p-B[i][0])/B[i][1];w+=B[i][2]*Math.exp(-d*d);}return w;}
function r06Render3D(fr){var c=document.getElementById('r06-3d');if(!c||!fr.vox)return;var ctx=c.getContext('2d'),W=c.width,H=c.height,cx=W/2,cy=H/2,scale=21;ctx.fillStyle='#000';ctx.fillRect(0,0,W,H);ctx.globalCompositeOperation='lighter';var sp=r06Sprites(),v=fr.vox,cr=Math.cos(r06Rot),sr=Math.sin(r06Rot),ct=Math.cos(r06Tilt),st=Math.sin(r06Tilt),jmax=v.jmax||1e-30;for(var i=0;i<v.n;i++){var p=v.p[i],X=v.x[i]*cr+v.z[i]*sr,Z=-v.x[i]*sr+v.z[i]*cr,Y=v.y[i]*ct-Z*st,u=cx+scale*X,vv=cy-scale*Y,a=Math.pow(Math.min(p,.3),.5),bd=r06Band(p),ph=.5+.5*Math.cos(8*v.ph[i]),w=.06*Math.pow(p,.3)+.40*bd*(.35+.65*ph),size=(.5+1*a+.8*bd)*16;if(u+size<0||vv+size<0||u-size>W||vv-size>H)continue;var b=((v.ph[i]+Math.PI)/(2*Math.PI)*12)|0;if(b>11)b=11;ctx.globalAlpha=Math.min(.85,w*.45);ctx.drawImage(sp.hue[b],u-size/2,vv-size/2,size,size);if(p>.5){ctx.globalAlpha=Math.min(.7,(p-.5)*1.1);ctx.drawImage(sp.white,u-size*.35,vv-size*.35,size*.7,size*.7);}var jx=v.jx[i],jy=v.jy[i],jz=v.jz[i],jm=Math.sqrt(jx*jx+jy*jy+jz*jz);if(jm>.05*jmax&&p>.003&&p<.7){var jXp=jx*cr+jz*sr,jZp=-jx*sr+jz*cr,jYp=jy*ct-jZp*st,nrm=Math.sqrt(jXp*jXp+jYp*jYp)+1e-12,L=4+13*(jm/jmax),du=jXp/nrm*L,dv=-jYp/nrm*L;ctx.globalAlpha=Math.min(.55,.12+.7*(jm/jmax));ctx.strokeStyle=sp.line[b];ctx.beginPath();ctx.moveTo(u-du,vv-dv);ctx.lineTo(u+du,vv+dv);ctx.stroke();}}
ctx.globalAlpha=1;ctx.globalCompositeOperation='source-over';ctx.beginPath();ctx.arc(cx,cy,3,0,2*Math.PI);ctx.fillStyle='#fff';ctx.fill();ctx.font='11px monospace';ctx.fillStyle='rgba(255,120,120,.95)';ctx.fillText('p+',cx+8,cy-7);
if(r06Mode==='light'){var F=fr.obs.field;ctx.strokeStyle='rgba(120,210,255,.85)';ctx.lineWidth=2;ctx.beginPath();for(var xx=20;xx<W-20;xx++){var yy=48+18*Math.sin((xx/W)*8*Math.PI-r06Eng.t*.8);if(xx===20)ctx.moveTo(xx,yy);else ctx.lineTo(xx,yy);}ctx.stroke();ctx.fillStyle='#8bd8ff';ctx.fillText('E(t) along z  '+F.toFixed(3),20,24);}
}
var r06Off=null;function r06Paint(id,mat,N){var c=document.getElementById(id);if(!c)return;if(!r06Off)r06Off=document.createElement('canvas');r06Off.width=N;r06Off.height=N;var o=r06Off.getContext('2d'),img=o.createImageData(N,N);for(var i=0;i<N;i++)for(var j=0;j<N;j++){var v=mat[i][N-1-j],p=(j*N+i)*4;img.data[p]=Math.min(255,v*3|0);img.data[p+1]=Math.min(255,v*1.5|0);img.data[p+2]=Math.min(255,v*.5|0);img.data[p+3]=255;}o.putImageData(img,0,0);c.width=c.height=220;var ctx=c.getContext('2d');ctx.imageSmoothingEnabled=true;ctx.drawImage(r06Off,0,0,220,220);}
function r06Reset(mode){if(r06Timer){clearInterval(r06Timer);r06Timer=null;}r06Playing=false;r06Mode=mode;r06Eng=new QGTHydrogenR06();r06Eng.setMode(mode);r06Frames=[r06Snapshot()];r06Hist=[r06Frames[0].obs];r06Fi=0;document.getElementById('r06-play').textContent='▶ Play';document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));r06Draw(0);}
function r06Draw(fi){if(!r06Eng)r06Reset('prepare');var max=R06_LIMITS[r06Mode];fi=Math.max(0,Math.min(fi,max));while(r06Frames.length<=fi){r06Eng.step();var s=r06Snapshot();r06Frames.push(s);r06Hist.push(s.obs);}r06Fi=fi;var fr=r06Frames[fi],sl=document.getElementById('r06-slider');sl.max=max;sl.value=fi;r06Rot+=.035;r06Render3D(fr);r06Paint('r06-xy',fr.xy,r06Eng.N);r06Paint('r06-xz',fr.xz,r06Eng.N);r06Paint('r06-yz',fr.yz,r06Eng.N);r06PlotHistory();var o=fr.obs,label=r06Mode==='prepare'?'imaginary τ':'τ';document.getElementById('r06-time').textContent=label+' = '+fr.t.toFixed(2);var info='';if(r06Mode==='prepare')info='<b>PREPARE</b> numerical eigenprojection · fidelity(1s)='+o.p1.toFixed(4)+' · <r>='+o.meanR.toFixed(3)+' · not physical capture';else if(r06Mode==='life')info='<b>LIVE</b> unitary bound superposition · <z>='+o.z.toFixed(3)+' · <r>='+o.meanR.toFixed(3)+' · norm drift='+(Math.abs(o.norm-1)).toExponential(1);else info='<b>LIGHT</b> H=H0−F(τ)z · F='+o.field.toFixed(4)+' · <z>='+o.z.toFixed(3)+' · P1s='+o.p1.toFixed(3)+' · P2p='+o.p2.toFixed(3)+' · Δnorm='+(Math.abs(o.norm-1)).toExponential(1);document.getElementById('r06-info').innerHTML=info;document.getElementById('r06-explain').innerHTML=R06_TEXT[r06Mode];}
function r06PlotHistory(){var c=document.getElementById('r06-history'),ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.fillStyle='#071018';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#31485e';ctx.strokeRect(35,10,W-45,H-30);var data=r06Hist.slice(0,r06Fi+1);if(data.length<2)return;var vals;if(r06Mode==='prepare')vals=data.map(o=>o.p1);else vals=data.map(o=>o.z);var mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals);if(mx-mn<1e-6){mx+=1;mn-=1;}ctx.strokeStyle=r06Mode==='prepare'?'#e6c35a':'#82d4ff';ctx.lineWidth=2;ctx.beginPath();vals.forEach((v,i)=>{var x=35+(W-45)*i/(Math.max(1,R06_LIMITS[r06Mode])),y=10+(H-30)*(1-(v-mn)/(mx-mn));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.fillStyle='#b9c8d7';ctx.font='12px monospace';ctx.fillText(r06Mode==='prepare'?'fidelity 1s':'<z>',4,18);if(r06Mode==='light'){var fields=data.map(o=>o.field),fa=Math.max(.001,...fields.map(Math.abs));ctx.strokeStyle='#f28e8e';ctx.beginPath();fields.forEach((v,i)=>{var x=35+(W-45)*i/R06_LIMITS.light,y=10+(H-30)*(.5-.42*v/fa);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.fillStyle='#f28e8e';ctx.fillText('E(t)',4,35);}}
function r06Toggle(){if(r06Playing){clearInterval(r06Timer);r06Timer=null;r06Playing=false;document.getElementById('r06-play').textContent='▶ Play';return;}r06Playing=true;document.getElementById('r06-play').textContent='⏸ Pause';r06Timer=setInterval(function(){var n=r06Fi+1;if(n>R06_LIMITS[r06Mode]){clearInterval(r06Timer);r06Timer=null;r06Playing=false;document.getElementById('r06-play').textContent='▶ Replay';return;}r06Draw(n);},150);}
var R06_TEXT={prepare:'<b>PREPARE.</b> A broad trial wavefunction is projected toward the Coulomb ground eigenstate by imaginary-time propagation. This is a numerical state-preparation/eigenprojection, <b>not</b> spontaneous radiative recombination.',life:'<b>LIVE.</b> A formed bound atom evolves unitarily as a coherent 1s+2p+3d superposition. Brightness is |ψ|², colour is local phase, filaments are probability current. There is no classical orbit.',light:'<b>LIGHT.</b> The atom starts in 1s and is driven by an explicit near-resonant electric pulse through V_light=−F(τ)z. The blue wave is the applied field; the graph tracks the induced dipole and the 1s/2p populations.'};
function r06Guided(){var seq=['prepare','life','light'],i=0;function go(){r06Reset(seq[i]);r06Toggle();var ms=(R06_LIMITS[seq[i]]+2)*150+1000;setTimeout(function(){i++;if(i<seq.length)go();},ms);}go();}
document.addEventListener('DOMContentLoaded',function(){document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>r06Reset(b.dataset.mode)));document.getElementById('r06-play').addEventListener('click',r06Toggle);document.getElementById('r06-reset').addEventListener('click',()=>r06Reset(r06Mode));document.getElementById('r06-slider').addEventListener('input',e=>r06Draw(+e.target.value));document.getElementById('r06-guided').addEventListener('click',r06Guided);var c=document.getElementById('r06-3d'),drag=false,lx=0,ly=0;c.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!drag)return;r06Rot+=(e.clientX-lx)*.012;r06Tilt=Math.max(.1,Math.min(2.6,r06Tilt+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;r06Render3D(r06Frames[r06Fi]);});c.addEventListener('pointerup',()=>drag=false);r06Reset('prepare');});
