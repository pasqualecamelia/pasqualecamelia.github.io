// QGT Hydrogen3D r07 — bilinear atom-light readout
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
  for(var a=1;a<N-1;a++)for(var b=1;b<N-1;b++)for(var c=1;c<N-1;c++){if(Math.abs(xs[a])>RCLIP||Math.abs(xs[b])>RCLIP||Math.abs(xs[c])>RCLIP)continue;var idx=a*N2+b*N+c;if(pA[idx]>thr&&n<count){var h1=Math.sin((idx+1)*12.9898)*43758.5453,h2=Math.sin((idx+1)*78.233)*12345.6789,h3=Math.sin((idx+1)*39.425)*24680.1357;h1=h1-Math.floor(h1);h2=h2-Math.floor(h2);h3=h3-Math.floor(h3);X[n]=xs[a]+(h1-.5)*2*JIT;Y[n]=xs[b]+(h2-.5)*2*JIT;Z[n]=xs[c]+(h3-.5)*2*JIT;P[n]=pA[idx]/pmax;PH[n]=Math.atan2(im[idx],re[idx]);var jx=re[idx]*(im[idx+N2]-im[idx-N2])-im[idx]*(re[idx+N2]-re[idx-N2]);var jy=re[idx]*(im[idx+N]-im[idx-N])-im[idx]*(re[idx+N]-re[idx-N]);var jz=re[idx]*(im[idx+1]-im[idx-1])-im[idx]*(re[idx+1]-re[idx-1]);JX[n]=jx;JY[n]=jy;JZ[n]=jz;var jm=Math.sqrt(jx*jx+jy*jy+jz*jz);if(jm>jmax)jmax=jm;n++;}}
  return {n:n,x:X,y:Y,z:Z,p:P,ph:PH,jx:JX,jy:JY,jz:JZ,jmax:jmax};
};
Engine.prototype.slices=function(){var N=this.N,N2=N*N,h=N>>1,re=this.re,im=this.im;function blank(){var m=[];for(var i=0;i<N;i++)m.push(new Float64Array(N));return m;}var xy=blank(),xz=blank(),yz=blank(),max=0,a,b;for(a=0;a<N;a++)for(b=0;b<N;b++){var i1=a*N2+b*N+h,i2=a*N2+h*N+b,i3=h*N2+a*N+b;xy[a][b]=re[i1]*re[i1]+im[i1]*im[i1];xz[a][b]=re[i2]*re[i2]+im[i2]*im[i2];yz[a][b]=re[i3]*re[i3]+im[i3]*im[i3];max=Math.max(max,xy[a][b],xz[a][b],yz[a][b]);}var sc=max>0?255/max:0;for(a=0;a<N;a++)for(b=0;b<N;b++){xy[a][b]=Math.round(xy[a][b]*sc);xz[a][b]=Math.round(xz[a][b]*sc);yz[a][b]=Math.round(yz[a][b]*sc);}return {xy:xy,xz:xz,yz:yz};};
global.QGTHydrogenR07=Engine;
})(typeof window!=='undefined'?window:globalThis);

// ---------- UI / renderer ----------
var r07Eng=null,r07Bi=null,r07Frames=[],r07Fi=0,r07Playing=false,r07Timer=null,r07Mode='prepare',r07Hist=[];
var R07_LIMITS={prepare:40,life:100,light:180,bilinear:70};
function r07Limit(){return R07_LIMITS[r07Mode];}
var R07_REF={beta11:1.5296328911991654e-3,alphaEff:4.521633076009848,alphaStatic:4.500003906449981,normDrift:1.5384360452230794e-12,noOverlap:6.800428699210772e-9};

function cMul(a,b){return {r:a.r*b.r-a.i*b.i,i:a.r*b.i+a.i*b.r};}
function cDiv(a,b){var d=b.r*b.r+b.i*b.i;return {r:(a.r*b.r+a.i*b.i)/d,i:(a.i*b.r-a.r*b.i)/d};}
function cArg(a){return Math.atan2(a.i,a.r);}

function BilinearEnsemble(){
  this.base=new QGTHydrogenR07();
  this.base.setMode('light');
  this.t=0;this.steps=0;this.eps=0.0040;this.T=10.0;this.delay=1.25;this.I12=0;this.nullControl=false;
  this.branches=[
    {s1:1,s2:1,re:new Float64Array(this.base.re),im:new Float64Array(this.base.im)},
    {s1:1,s2:-1,re:new Float64Array(this.base.re),im:new Float64Array(this.base.im)},
    {s1:-1,s2:1,re:new Float64Array(this.base.re),im:new Float64Array(this.base.im)},
    {s1:-1,s2:-1,re:new Float64Array(this.base.re),im:new Float64Array(this.base.im)}
  ];
}
BilinearEnsemble.prototype._sin2=function(t,T,d){var x=(t-d)/T;if(x<0||x>1)return 0;var q=Math.sin(Math.PI*x);return q*q;};
BilinearEnsemble.prototype.field=function(s1,s2,t){var f1=this._sin2(t,this.T,0),f2=this.nullControl?0:this._sin2(t,this.T,this.delay);return this.eps*(s1*f1+s2*f2);};
BilinearEnsemble.prototype.step=function(){var self=this,t0=this.t,dt=this.base.cfg.dt,tm=t0+dt/2;this.I12+=this.nullControl?0:this._sin2(tm,this.T,0)*this._sin2(tm,this.T,this.delay)*dt;
  this.branches.forEach(function(br){self.base.re=br.re;self.base.im=br.im;self.base.t=t0;self.base.mode='bilinear';self.base.lightField=function(){return self.field(br.s1,br.s2,self.base.t);};self.base._realStep();br.re=self.base.re;br.im=self.base.im;});
  this.t=t0+dt;this.steps++;
};
BilinearEnsemble.prototype._amp=function(br){var rr=0,ii=0,b=this.base.b1,re=br.re,im=br.im;for(var i=0;i<b.length;i++){rr+=b[i]*re[i];ii+=b[i]*im[i];}return {r:rr,i:ii};};
BilinearEnsemble.prototype.beta=function(){var A=this.branches.map(this._amp.bind(this)),num=cMul(A[0],A[3]),den=cMul(A[1],A[2]),ratio=cDiv(num,den);var beta=.25*cArg(ratio),alpha=this.I12>1e-14?beta/(this.eps*this.eps*this.I12):0;return {beta11:beta,alphaEff:alpha,amps:A,ratio:ratio};};
BilinearEnsemble.prototype._setMain=function(){var br=this.branches[0];this.base.re=br.re;this.base.im=br.im;this.base.t=this.t;this.base.mode='bilinear';this.base.lightField=this.field.bind(this,1,1);};
BilinearEnsemble.prototype.observables=function(){this._setMain();var o=this.base.observables(),b=this.beta();o.field1=this.eps*this._sin2(this.t,this.T,0);o.field2=this.nullControl?0:this.eps*this._sin2(this.t,this.T,this.delay);o.field=o.field1+o.field2;o.beta11=b.beta11;o.alphaEff=b.alphaEff;o.I12=this.I12;o.eps=this.eps;o.nullControl=this.nullControl;return o;};
BilinearEnsemble.prototype.voxels=function(n){this._setMain();return this.base.voxels(n);};
BilinearEnsemble.prototype.slices=function(){this._setMain();return this.base.slices();};
BilinearEnsemble.prototype.mixedXZ=function(){var N=this.base.N,N2=N*N,h=N>>1,m=[],max=0;for(var a=0;a<N;a++){var row=new Float64Array(N);for(var c=0;c<N;c++){var idx=a*N2+h*N+c,val=0;for(var q=0;q<4;q++){var br=this.branches[q],rho=br.re[idx]*br.re[idx]+br.im[idx]*br.im[idx],sgn=(q===0||q===3)?1:-1;val+=sgn*rho;}val/=4*this.eps*this.eps;row[c]=val;max=Math.max(max,Math.abs(val));}m.push(row);}return {mat:m,max:max||1};};

function r07Snapshot(){
  if(r07Mode==='bilinear'){
    var s=r07Bi.slices(),mx=r07Bi.mixedXZ();
    return {t:r07Bi.t,obs:r07Bi.observables(),xy:s.xy,xz:s.xz,mixed:mx.mat,mixedMax:mx.max,vox:r07Bi.voxels(20000)};
  }
  var s=r07Eng.slices();return {t:r07Eng.t,obs:r07Eng.observables(),xy:s.xy,xz:s.xz,yz:s.yz,vox:r07Eng.voxels(20000)};
}

var r07Rot=.25,r07Tilt=1.15,r07SpritesCache=null;
function r07Sprites(){if(r07SpritesCache)return r07SpritesCache;function mk(h,w){var c=document.createElement('canvas');c.width=c.height=48;var g=c.getContext('2d'),col=w?'rgba(255,255,255,':'hsla('+h+',95%,62%,';var gr=g.createRadialGradient(24,24,0,24,24,24);gr.addColorStop(0,col+'1)');gr.addColorStop(.22,col+'.75)');gr.addColorStop(.55,col+'.18)');gr.addColorStop(1,col+'0)');g.fillStyle=gr;g.fillRect(0,0,48,48);return c;}var arr=[],line=[];for(var i=0;i<16;i++){arr.push(mk(195+i*22,false));line.push('hsl('+(195+i*22)+',95%,72%)');}return r07SpritesCache={hue:arr,white:mk(0,true),line:line,orange:mk(28,false),blue:mk(205,false)};}
function r07Band(p){var w=0,B=[[.45,.07,.7],[.13,.03,1],[.035,.010,1.4],[.008,.0028,1.1]];for(var i=0;i<4;i++){var d=(p-B[i][0])/B[i][1];w+=B[i][2]*Math.exp(-d*d);}return w;}
function r07Wave(ctx,W,H,y0,amp,phase,color,label,alignRight){ctx.save();ctx.globalCompositeOperation='lighter';ctx.lineWidth=2.4;ctx.strokeStyle=color;ctx.shadowBlur=12;ctx.shadowColor=color;ctx.beginPath();for(var x=0;x<W;x++){var env=Math.exp(-Math.pow((x-W*.5)/(W*.22),2)),y=y0+amp*env*Math.sin(x*.055-phase);if(x===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();ctx.shadowBlur=0;ctx.globalCompositeOperation='source-over';ctx.fillStyle=color;ctx.font='11px monospace';ctx.fillText(label,alignRight?W-175:20,y0-22);ctx.restore();}
function r07Render3D(fr){var c=document.getElementById('r07-3d');if(!c||!fr.vox)return;var ctx=c.getContext('2d'),W=c.width,H=c.height,cx=W/2,cy=H/2,scale=Math.min(W/46,H/26);var bg=ctx.createRadialGradient(cx,cy,10,cx,cy,Math.max(W,H)*.62);bg.addColorStop(0,'#06152a');bg.addColorStop(.45,'#020914');bg.addColorStop(1,'#00040a');ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
  // subtle spatial grid
  ctx.strokeStyle='rgba(90,150,210,.055)';ctx.lineWidth=1;for(var gx=0;gx<W;gx+=64){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke();}for(var gy=0;gy<H;gy+=64){ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();}
  ctx.globalCompositeOperation='lighter';var sp=r07Sprites(),v=fr.vox,cr=Math.cos(r07Rot),sr=Math.sin(r07Rot),ct=Math.cos(r07Tilt),st=Math.sin(r07Tilt),jmax=v.jmax||1e-30;
  for(var i=0;i<v.n;i++){var p=v.p[i],X=v.x[i]*cr+v.z[i]*sr,Z=-v.x[i]*sr+v.z[i]*cr,Y=v.y[i]*ct-Z*st,u=cx+scale*X,vv=cy-scale*Y,a=Math.pow(Math.min(p,.3),.5),bd=r07Band(p),ph=.5+.5*Math.cos(8*v.ph[i]),w=.045*Math.pow(p,.28)+.48*bd*(.28+.72*ph),size=(.55+1.25*a+.9*bd)*18;if(u+size<0||vv+size<0||u-size>W||vv-size>H)continue;var b=((v.ph[i]+Math.PI)/(2*Math.PI)*16)|0;if(b>15)b=15;ctx.globalAlpha=Math.min(.75,w*.46);ctx.drawImage(sp.hue[b],u-size/2,vv-size/2,size,size);if(p>.45){ctx.globalAlpha=Math.min(.52,(p-.45));ctx.drawImage(sp.white,u-size*.3,vv-size*.3,size*.6,size*.6);}var jx=v.jx[i],jy=v.jy[i],jz=v.jz[i],jm=Math.sqrt(jx*jx+jy*jy+jz*jz);if(jm>.06*jmax&&p>.002&&p<.72){var jXp=jx*cr+jz*sr,jZp=-jx*sr+jz*cr,jYp=jy*ct-jZp*st,nrm=Math.sqrt(jXp*jXp+jYp*jYp)+1e-12,L=5+15*(jm/jmax),du=jXp/nrm*L,dv=-jYp/nrm*L;ctx.globalAlpha=Math.min(.48,.10+.62*(jm/jmax));ctx.strokeStyle=sp.line[b];ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(u-du,vv-dv);ctx.quadraticCurveTo(u,vv-2,u+du,vv+dv);ctx.stroke();}}
  // nucleus glow
  ctx.globalAlpha=1;var ng=ctx.createRadialGradient(cx,cy,0,cx,cy,24);ng.addColorStop(0,'rgba(255,255,235,1)');ng.addColorStop(.2,'rgba(255,188,78,.95)');ng.addColorStop(.55,'rgba(255,96,24,.35)');ng.addColorStop(1,'rgba(255,90,20,0)');ctx.fillStyle=ng;ctx.beginPath();ctx.arc(cx,cy,24,0,2*Math.PI);ctx.fill();ctx.fillStyle='#ffe9a8';ctx.beginPath();ctx.arc(cx,cy,3.5,0,2*Math.PI);ctx.fill();ctx.globalCompositeOperation='source-over';ctx.fillStyle='#ffd49a';ctx.font='11px monospace';ctx.fillText('p+',cx+10,cy-8);
  if(r07Mode==='light'){r07Wave(ctx,W,H,52,19,(r07Eng?r07Eng.t:0)*.8,'rgba(118,215,255,.9)','near-resonant E(t) · dipole probe',false);}
  if(r07Mode==='bilinear'){var o=fr.obs,ph=fr.t*.35;r07Wave(ctx,W,H,64,18,ph,'rgba(255,153,76,.9)','probe 1 · s₁',false);r07Wave(ctx,W,H,H-60,18,-ph,'rgba(115,198,255,.9)','probe 2 · s₂',true);ctx.fillStyle='rgba(225,235,245,.92)';ctx.font='12px monospace';ctx.fillText('four coherent branches: ++  +−  −+  −−',20,H-18);}
}

var r07Off=null;
function r07PaintDensity(id,mat,N){var c=document.getElementById(id);if(!c)return;if(!r07Off)r07Off=document.createElement('canvas');r07Off.width=N;r07Off.height=N;var o=r07Off.getContext('2d'),img=o.createImageData(N,N);for(var i=0;i<N;i++)for(var j=0;j<N;j++){var v=mat[i][N-1-j]/255,p=(j*N+i)*4;v=Math.pow(Math.max(0,v),.42);var r=Math.min(255,255*Math.pow(v,.65)),g=Math.min(255,210*Math.pow(v,1.2)),b=Math.min(255,255*(1-v)*.8+80*v);img.data[p]=r;img.data[p+1]=g;img.data[p+2]=b;img.data[p+3]=255;}o.putImageData(img,0,0);c.width=360;c.height=220;var ctx=c.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,c.width,c.height);ctx.imageSmoothingEnabled=true;ctx.drawImage(r07Off,0,0,c.width,c.height);}
function r07PaintMixed(id,mat,N,max){var c=document.getElementById(id);if(!c)return;if(!r07Off)r07Off=document.createElement('canvas');r07Off.width=N;r07Off.height=N;var o=r07Off.getContext('2d'),img=o.createImageData(N,N),m=max||1;for(var i=0;i<N;i++)for(var j=0;j<N;j++){var x=Math.max(-1,Math.min(1,mat[i][N-1-j]/m)),p=(j*N+i)*4,a=Math.abs(x);if(x>=0){img.data[p]=255;img.data[p+1]=Math.round(245*(1-a));img.data[p+2]=Math.round(245*(1-a));}else{img.data[p]=Math.round(235*(1-a));img.data[p+1]=Math.round(245*(1-a));img.data[p+2]=255;}img.data[p+3]=255;}o.putImageData(img,0,0);c.width=360;c.height=220;var ctx=c.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,c.width,c.height);ctx.imageSmoothingEnabled=true;ctx.drawImage(r07Off,0,0,c.width,c.height);}

function r07Reset(mode){if(r07Timer){clearInterval(r07Timer);r07Timer=null;}r07Playing=false;r07Mode=mode;r07Frames=[];r07Hist=[];r07Fi=0;if(mode==='bilinear'){r07Bi=new BilinearEnsemble();r07Eng=null;}else{r07Bi=null;r07Eng=new QGTHydrogenR07();r07Eng.setMode(mode);}r07Frames=[r07Snapshot()];r07Hist=[r07Frames[0].obs];document.getElementById('r07-play').textContent='▶ Play';document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));r07Draw(0);}
function r07Draw(fi){var max=r07Limit();fi=Math.max(0,Math.min(fi,max));while(r07Frames.length<=fi){if(r07Mode==='bilinear')r07Bi.step();else r07Eng.step();var s=r07Snapshot();r07Frames.push(s);r07Hist.push(s.obs);}r07Fi=fi;var fr=r07Frames[fi],sl=document.getElementById('r07-slider');sl.max=max;sl.value=fi;r07Rot+=.028;r07Render3D(fr);r07PaintDensity('r07-xy',fr.xy,(r07Mode==='bilinear'?r07Bi.base:r07Eng).N);r07PaintDensity('r07-xz',fr.xz,(r07Mode==='bilinear'?r07Bi.base:r07Eng).N);if(r07Mode==='bilinear'){r07PaintMixed('r07-yz',fr.mixed,(r07Bi.base).N,fr.mixedMax);document.getElementById('r07-third-label').textContent='Walsh 11 · mixed XZ response';}else{r07PaintDensity('r07-yz',fr.yz,r07Eng.N);document.getElementById('r07-third-label').textContent='yz · x=0';}r07PlotHistory();var o=fr.obs,label=r07Mode==='prepare'?'imaginary τ':'τ';document.getElementById('r07-time').textContent=label+' = '+fr.t.toFixed(2);var info='';if(r07Mode==='prepare')info='<b>PREPARE</b> numerical eigenprojection · fidelity(1s)='+o.p1.toFixed(4)+' · ⟨r⟩='+o.meanR.toFixed(3)+' · not physical capture';else if(r07Mode==='life')info='<b>LIVE</b> unitary bound superposition · ⟨z⟩='+o.z.toFixed(3)+' · ⟨r⟩='+o.meanR.toFixed(3)+' · norm drift='+(Math.abs(o.norm-1)).toExponential(1);else if(r07Mode==='light')info='<b>LIGHT</b> H=H₀−F(τ)z · F='+o.field.toFixed(4)+' · ⟨z⟩='+o.z.toFixed(3)+' · P₁s='+o.p1.toFixed(3)+' · P₂p='+o.p2.toFixed(3)+' · Δnorm='+(Math.abs(o.norm-1)).toExponential(1);else info='<b>BILINEAR / WALSH 11</b> four coherent TDSE branches · β₁₁='+o.beta11.toExponential(4)+' rad · αeff(live)='+(o.I12>1e-6?o.alphaEff.toFixed(4):'—')+' · I₁₂='+o.I12.toFixed(3)+' · ε='+o.eps.toFixed(4)+(o.nullControl?' · NULL CONTROL: PROBE 2 OFF':'');document.getElementById('r07-info').innerHTML=info;document.getElementById('r07-explain').innerHTML=R07_TEXT[r07Mode];r07UpdateMetrics(o);}
function r07UpdateMetrics(o){var set=function(id,v){var e=document.getElementById(id);if(e)e.textContent=v;};set('m-beta',r07Mode==='bilinear'?o.beta11.toExponential(3)+' rad':R07_REF.beta11.toExponential(3)+' rad');set('m-alpha',r07Mode==='bilinear'&&o.I12>1e-6?o.alphaEff.toFixed(4):R07_REF.alphaStatic.toFixed(6));set('m-p1',o.p1!=null?o.p1.toFixed(4):'—');set('m-p2',o.p2!=null?o.p2.toFixed(4):'—');set('m-norm',o.norm!=null?o.norm.toFixed(9):'—');}
function r07PlotHistory(){var c=document.getElementById('r07-history'),ctx=c.getContext('2d'),W=c.width,H=c.height;ctx.fillStyle='#071018';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#31485e';ctx.strokeRect(44,10,W-54,H-30);var data=r07Hist.slice(0,r07Fi+1);if(data.length<2)return;var vals,label,color;if(r07Mode==='prepare'){vals=data.map(o=>o.p1);label='fidelity 1s';color='#e6c35a';}else if(r07Mode==='bilinear'){vals=data.map(o=>o.beta11);label='β11';color='#c99cff';}else{vals=data.map(o=>o.z);label='⟨z⟩';color='#82d4ff';}var mn=Math.min.apply(null,vals),mx=Math.max.apply(null,vals);if(mx-mn<1e-12){mx+=1e-6;mn-=1e-6;}ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();vals.forEach((v,i)=>{var x=44+(W-54)*i/(Math.max(1,r07Limit())),y=10+(H-30)*(1-(v-mn)/(mx-mn));if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.fillStyle='#b9c8d7';ctx.font='11px monospace';ctx.fillText(label,4,18);if(r07Mode==='bilinear'){var f1=data.map(o=>o.field1||0),f2=data.map(o=>o.field2||0),fa=Math.max(.001,...f1.concat(f2).map(Math.abs));[['#ff9d4d',f1],['#75c6ff',f2]].forEach(function(ent){ctx.strokeStyle=ent[0];ctx.lineWidth=1;ctx.beginPath();ent[1].forEach((v,i)=>{var x=44+(W-54)*i/r07Limit(),y=10+(H-30)*(.5-.38*v/fa);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();});}}
function r07Toggle(){if(r07Playing){clearInterval(r07Timer);r07Timer=null;r07Playing=false;document.getElementById('r07-play').textContent='▶ Play';return;}r07Playing=true;document.getElementById('r07-play').textContent='⏸ Pause';var delay=r07Mode==='bilinear'?220:150;r07Timer=setInterval(function(){var n=r07Fi+1;if(n>r07Limit()){clearInterval(r07Timer);r07Timer=null;r07Playing=false;document.getElementById('r07-play').textContent='▶ Replay';return;}r07Draw(n);},delay);}
var R07_TEXT={prepare:'<b>PREPARE.</b> A broad trial wavefunction is projected toward the Coulomb ground eigenstate by imaginary-time propagation. This is numerical eigenstate preparation, <b>not</b> spontaneous radiative capture.',life:'<b>LIVE.</b> A formed bound atom evolves unitarily as a coherent 1s+2p+3d superposition. Brightness is |ψ|², hue is local phase, filaments indicate probability current. There is no classical orbit.',light:'<b>LIGHT.</b> The atom starts in 1s and is driven by an explicit near-resonant electric field through V=−F(τ)z. This is the ordinary Schrödinger-grade atom–light sector.',bilinear:'<b>BILINEAR / WALSH 11.</b> Two weak reversible probes are applied with independent signs. The browser evolves the four TDSE branches (++,+−,−+,−−) and isolates the mixed phase channel β₁₁ = ¼ Arg[(M++M--)/(M+-M-+)]. The third slice is the sign-normalized mixed density response. The high-precision benchmark below comes from the independent partial-wave solver, not from fitting this browser animation.'};
function r07Guided(){var seq=['prepare','life','light','bilinear'],i=0;function go(){r07Reset(seq[i]);r07Toggle();var ms=(r07Limit()+2)*(seq[i]==='bilinear'?220:150)+900;setTimeout(function(){i++;if(i<seq.length)go();},ms);}go();}
function r07ToggleNoOverlap(){var flag=(r07Mode==='bilinear'&&r07Bi)?!r07Bi.nullControl:true;if(r07Timer){clearInterval(r07Timer);r07Timer=null;}r07Playing=false;r07Mode='bilinear';r07Eng=null;r07Bi=new BilinearEnsemble();r07Bi.nullControl=flag;r07Frames=[r07Snapshot()];r07Hist=[r07Frames[0].obs];r07Fi=0;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode==='bilinear'));document.getElementById('r07-control').classList.toggle('active',flag);document.getElementById('r07-play').textContent='▶ Play';r07Draw(0);}
if(typeof globalThis!=='undefined')globalThis.QGTBilinearR07=BilinearEnsemble;
if(typeof document!=='undefined')document.addEventListener('DOMContentLoaded',function(){document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>r07Reset(b.dataset.mode)));document.getElementById('r07-play').addEventListener('click',r07Toggle);document.getElementById('r07-reset').addEventListener('click',()=>r07Reset(r07Mode));document.getElementById('r07-slider').addEventListener('input',e=>r07Draw(+e.target.value));document.getElementById('r07-guided').addEventListener('click',r07Guided);document.getElementById('r07-control').addEventListener('click',r07ToggleNoOverlap);var c=document.getElementById('r07-3d'),drag=false,lx=0,ly=0;c.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!drag)return;r07Rot+=(e.clientX-lx)*.012;r07Tilt=Math.max(.1,Math.min(2.6,r07Tilt+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;r07Render3D(r07Frames[r07Fi]);});c.addEventListener('pointerup',()=>drag=false);r07Reset('prepare');});
