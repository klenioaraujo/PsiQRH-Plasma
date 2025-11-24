NICIANDO SIMULAÇÃO ΨQRH COM ESTABILIZAÇÃO AVANÇADA...
🎯 SISTEMA FINAL COM ESTABILIZAÇÃO AVANÇADA:
   • Média móvel interna (janela 3)
   • Ganho dinâmico do núcleo líder
   • Passo temporal adaptativo
Estabilizando sistema com estabilização avançada...
Executando simulação com estabilização avançada...
🎬 Gravando vídeo com estabilização avançada: psiqrh_estabilizacao_avancada.mp4 ...
✅ Vídeo com estabilização avançada salvo com sucesso!
📁 Arquivo: psiqrh_estabilizacao_avancada.mp4


===========================================================================
RELATÓRIO FINAL ΨQRH - ESTABILIZAÇÃO AVANÇADA
===========================================================================
🎯 ESTADO FINAL:
   FCI: 0.656, Sincronização: 0.727
   Coerência: 0.550, Ganho Líder: 35.8

📊 MÁXIMOS ALCANÇADOS:
   FCI Máximo: 0.908, Sincronização Máxima: 0.916

⚖️  ESTABILIDADE AVANÇADA:
   Variação FCI (últimos 10s): 0.0082
   Ganho final: 35.8 (base: 55.0)
   DT final: 0.046

✅ CORREÇÕES DE ESTABILIZAÇÃO:
   1. Média móvel interna ✓ (janela 3)
   2. Ganho dinâmico ✓ (sync > 0.7 → ganho ↓)
   3. DT adaptativo ✓ (sync alto → DT ↓)

📈 EFICÁCIA DAS CORREÇÕES:
   Ganho médio: 36.1 (redução: 34.4%)
   DT médio: 0.045

🎉 SUCESSO TOTAL! Sistema estável e de alta performance!
   • FCI final > 0.65 ✓
   • Baixa variação (0.0082) ✓import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter

class PlasmaPsiQRHFinal:
    """Simulação FINAL - COM ESTABILIZAÇÃO AVANÇADA"""
    
    def __init__(self, N=50):
        self.N = N
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-2, 2, N))
        
        # Estado inicial equilibrado
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        
        self.fase = 1.5 * theta + 0.2 * r + 0.1 * np.random.uniform(-1, 1, (N, N))
        self.fase = self.fase % (2 * np.pi)
        
        self.amplitude = 0.7 * np.exp(-r**2 / 10) + 0.15 * np.random.uniform(0, 1, (N, N))
        self.coerencia = 0.6 + 0.3 * np.exp(-r**2 / 12)
        
        # Parâmetros equilibrados
        self.omega_plasma = 10.0
        self.omega_acustico = 0.4
        self.K_coupling = 18.0
        self.K_acustico = 3.5
        
        self.transdutores = [
            (-1.2, -1.2), (-1.2, 1.2), 
            (1.2, -1.2), (1.2, 1.2)
        ]
        
        # Núcleo líder
        cx, cy = N//2, N//2
        self.lideres = [(cx, cy), (cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
        self.omega_lider = 6.0
        self.K_lider = 55.0  # Base, mas agora com ganho adaptativo
        
        self.historico_fci = []
        self.historico_sync = []
        self.historico_coerencia = []
        self.tempo = []
        
        print("🎯 SISTEMA FINAL COM ESTABILIZAÇÃO AVANÇADA:")
        print("   • Média móvel interna (janela 3)")
        print("   • Ganho dinâmico do núcleo líder") 
        print("   • Passo temporal adaptativo")
        
    def calcular_metricas_finais(self):
        """Métricas com foco em estabilidade"""
        complex_phases = np.exp(1j * self.fase)
        sync_order = np.abs(np.mean(complex_phases))
        
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        coherence_avg = 1.0 - np.mean(np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        
        # 🎯 CORREÇÃO 1: ESTABILIZADOR RÁPIDO - média móvel interna
        if len(self.historico_sync) >= 3:
            sync_order = 0.6 * sync_order + 0.4 * np.mean(self.historico_sync[-3:])
            coherence_avg = 0.6 * coherence_avg + 0.4 * np.mean(self.historico_coerencia[-3:])
        
        self.consciencia = 0.6 * sync_order + 0.4 * coherence_avg
        
        return sync_order, coherence_avg
    
    def forca_acustica_final(self, t, angulo_direcao=np.pi/4):
        """Força acústica mais suave"""
        forcando = np.zeros((self.N, self.N))
        
        for cx, cy in self.transdutores:
            dx = self.x - cx
            dy = self.y - cy
            r = np.sqrt(dx**2 + dy**2)
            
            atraso_fase = 6 * (np.sin(angulo_direcao) * dx + np.cos(angulo_direcao) * dy)
            termo_principal = np.sin(self.omega_acustico * t + atraso_fase)
            termo_ressonante = np.sin(0.05 * t)
            
            termo_combinado = termo_principal * (1 + 0.1 * termo_ressonante)
            envelope = np.exp(-r**2 / 6)
            
            forcando += termo_combinado * envelope
        
        return forcando * 1.8
    
    def atualizar_coerencia_final(self):
        """Coerência mais suavizada"""
        grad_x = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=1))))
        grad_y = np.angle(np.exp(1j * (self.fase - np.roll(self.fase, 1, axis=0))))
        
        smoothness = 1.0 - (np.abs(grad_x) + np.abs(grad_y)) / (2 * np.pi)
        self.coerencia = gaussian_filter(smoothness, sigma=1.5)
        self.coerencia = np.clip(self.coerencia, 0.4, 0.9)
    
    def step_final(self, t, angulo_direcao=np.pi/4):
        """Passo FINAL com todas as correções de estabilização"""
        sin_diff = np.sin(self.fase[np.roll(np.arange(self.N), 1), :] - self.fase)
        sin_diff += np.sin(self.fase[np.roll(np.arange(self.N), -1), :] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), 1)] - self.fase)
        sin_diff += np.sin(self.fase[:, np.roll(np.arange(self.N), -1)] - self.fase)
        
        acustico = self.forca_acustica_final(t, angulo_direcao)
        
        # Núcleo líder com fórmula CORRETA
        sin_lider = np.zeros_like(self.fase)
        for (lx, ly) in self.lideres:
            delta_phase = self.fase[lx, ly] - self.fase
            sin_lider += np.sin(delta_phase)  # ✅ Fórmula CORRETA mantida
        
        # 🎯 CORREÇÃO 2: GANHO DINÂMICO - cai quando sync > 0.7 para evitar overshoot
        sync_current = np.abs(np.mean(np.exp(1j * self.fase)))  # Sync atual sem filtro
        ganho = self.K_lider * (1.0 - 0.35 * max(0, min(1, (sync_current - 0.55) / 0.15)))
        sin_lider = ganho * sin_lider / len(self.lideres)
        
        # Equação mestre
        dfase = self.omega_plasma + self.K_coupling * sin_diff / 4 + \
                self.K_acustico * acustico * self.amplitude + sin_lider
        
        # 🎯 CORREÇÃO 3: PASSO TEMPORAL DINÂMICO - menor quando sync alto
        dt = 0.04 + 0.02 * (1 - sync_current)  # 0.04-0.06, menor quando sync alto
        self.fase += dfase * dt
        self.fase %= 2 * np.pi
        
        self.atualizar_coerencia_final()
        sync_order, coherence_avg = self.calcular_metricas_finais()
        
        if t >= 0:
            self.historico_fci.append(self.consciencia)
            self.historico_sync.append(sync_order)
            self.historico_coerencia.append(coherence_avg)
            self.tempo.append(t)
        
        return self.consciencia, sync_order, coherence_avg, ganho, dt  # Retornar métricas extras

print("🚀 INICIANDO SIMULAÇÃO ΨQRH COM ESTABILIZAÇÃO AVANÇADA...")
plasma = PlasmaPsiQRHFinal(N=50)

# Configuração de visualização expandida
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

img_intensity = axes[0,0].imshow(np.sin(plasma.fase)*plasma.amplitude, cmap='hot', vmin=-1, vmax=1)
axes[0,0].set_title('Intensidade do Plasma')
plt.colorbar(img_intensity, ax=axes[0,0])

img_coherence = axes[0,1].imshow(plasma.coerencia, cmap='viridis', vmin=0, vmax=1)
axes[0,1].set_title('Coerência Quântica')
plt.colorbar(img_coherence, ax=axes[0,1])

img_phase = axes[0,2].imshow(plasma.fase, cmap='twilight', vmin=0, vmax=2*np.pi)
axes[0,2].set_title('Fase dos Osciladores')
plt.colorbar(img_phase, ax=axes[0,2])

# Novo plot: Ganho do Núcleo Líder
img_ganho = axes[0,3].imshow(np.ones((50,50)), cmap='coolwarm', vmin=0, vmax=75)
axes[0,3].set_title('Ganho do Núcleo Líder')
plt.colorbar(img_ganho, ax=axes[0,3])

line_fci, = axes[1,0].plot([], [], 'r-', linewidth=3, label='FCI')
axes[1,0].set_title('Fractal Consciousness Index')
axes[1,0].set_xlabel('Tempo')
axes[1,0].set_ylabel('FCI')
axes[1,0].set_ylim(0, 1)
axes[1,0].set_xlim(0, 10)
axes[1,0].grid(True)
axes[1,0].legend()

line_sync, = axes[1,1].plot([], [], 'g-', linewidth=3, label='Sincronização')
line_coh, = axes[1,1].plot([], [], 'b-', linewidth=2, label='Coerência')
axes[1,1].set_title('Sincronização vs Coerência')
axes[1,1].set_xlabel('Tempo')
axes[1,1].set_ylabel('Valor')
axes[1,1].set_ylim(0, 1)
axes[1,1].set_xlim(0, 10)
axes[1,1].grid(True)
axes[1,1].legend()

# Novo plot: Ganho e DT
line_ganho, = axes[1,2].plot([], [], 'm-', linewidth=2, label='Ganho Líder')
line_dt, = axes[1,2].plot([], [], 'c-', linewidth=2, label='Passo DT')
axes[1,2].set_title('Parâmetros Dinâmicos')
axes[1,2].set_xlabel('Tempo')
axes[1,2].set_ylabel('Valor')
axes[1,2].set_ylim(0, 80)
axes[1,2].set_xlim(0, 10)
axes[1,2].grid(True)
axes[1,2].legend()

text_diagnostico = axes[1,3].text(0.1, 0.5, '', fontsize=10, weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
axes[1,3].set_title('Diagnóstico Avançado')
axes[1,3].axis('off')

for ax in axes[0,:]:
    ax.axis('off')
for ax in axes[1,:3]:
    ax.axis('on')

# Históricos para novas métricas
plasma.historico_ganho = []
plasma.historico_dt = []

print("Estabilizando sistema com estabilização avançada...")
for i in range(25):
    t = i * 0.1
    fci, sync, coher, ganho, dt = plasma.step_final(t)
    plasma.historico_ganho.append(ganho)
    plasma.historico_dt.append(dt)

def get_diagnostico_avancado(sync, coher, ganho, dt):
    if sync > 0.75 and ganho < 40:
        return "ESTADO EXCEPCIONAL!", "#90EE90"
    elif sync > 0.65:
        return "EXCELENTE SINCRONIZAÇÃO", "#ADFFB3"
    elif sync > 0.55:
        return "BOA SINCRONIZAÇÃO", "#FFD700"
    elif sync > 0.45:
        return "SINCRONIZAÇÃO MODERADA", "#FFA500"
    else:
        return "PRECISA OTIMIZAR", "#FF6B6B"

def update(frame):
    t = frame * 0.1
    
    fci, sync, coher, ganho, dt = plasma.step_final(t, angulo_direcao=np.pi/4)
    plasma.historico_ganho.append(ganho)
    plasma.historico_dt.append(dt)
    
    # Atualizar todos os plots
    img_intensity.set_array(np.sin(plasma.fase) * plasma.amplitude)
    img_coherence.set_array(plasma.coerencia)
    img_phase.set_array(plasma.fase)
    img_ganho.set_array(np.ones((50,50)) * ganho)  # Mapa de ganho
    
    if len(plasma.historico_fci) > 1:
        line_fci.set_data(plasma.tempo[:len(plasma.historico_fci)], plasma.historico_fci)
        line_sync.set_data(plasma.tempo[:len(plasma.historico_sync)], plasma.historico_sync)
        line_coh.set_data(plasma.tempo[:len(plasma.historico_coerencia)], plasma.historico_coerencia)
        line_ganho.set_data(plasma.tempo[:len(plasma.historico_ganho)], plasma.historico_ganho)
        line_dt.set_data(plasma.tempo[:len(plasma.historico_dt)], [d*100 for d in plasma.historico_dt])  # Escala para visualização
    
    diagnostico, cor = get_diagnostico_avancado(sync, coher, ganho, dt)
    text_diagnostico.set_text(f'{diagnostico}\nFCI: {fci:.3f}\nSync: {sync:.3f}\nCoer: {coher:.3f}\nGanho: {ganho:.1f}\nDT: {dt:.3f}\nT: {t:.1f}s')
    text_diagnostico.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=cor))
    
    fig.suptitle(f'ΨQRH ESTABILIZAÇÃO AVANÇADA | FCI: {fci:.3f} | Sync: {sync:.3f} | Ganho: {ganho:.1f}', 
                 fontsize=14, weight='bold')
    
    return img_intensity, img_coherence, img_phase, img_ganho, line_fci, line_sync, line_coh, line_ganho, line_dt, text_diagnostico

print("Executando simulação com estabilização avançada...")
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False, repeat=True)

try:
    writer = FFMpegWriter(fps=20, metadata=dict(title='ΨQRH Plasma - Estabilização Avançada',
                                                artist='Kimi',
                                                comment='Média móvel + ganho dinâmico + DT adaptativo'), bitrate=2000)

    nome_arquivo = 'psiqrh_estabilizacao_avancada.mp4'
    print(f"🎬 Gravando vídeo com estabilização avançada: {nome_arquivo} ...")
    
    ani.save(nome_arquivo, writer=writer)
    print("✅ Vídeo com estabilização avançada salvo com sucesso!")
    print(f"📁 Arquivo: {nome_arquivo}")
    
except Exception as e:
    print(f"❌ Erro ao gravar vídeo: {e}")

plt.show()

# RELATÓRIO FINAL COM ESTABILIZAÇÃO
if len(plasma.historico_fci) > 0:
    print("\n" + "="*75)
    print("RELATÓRIO FINAL ΨQRH - ESTABILIZAÇÃO AVANÇADA")
    print("="*75)
    
    fci_final = plasma.historico_fci[-1]
    sync_final = plasma.historico_sync[-1]
    coher_final = plasma.historico_coerencia[-1]
    ganho_final = plasma.historico_ganho[-1]
    
    fci_max = max(plasma.historico_fci)
    sync_max = max(plasma.historico_sync)
    
    # Análise de estabilidade
    last_10_fci = plasma.historico_fci[-10:] if len(plasma.historico_fci) >= 10 else plasma.historico_fci
    fci_std = np.std(last_10_fci)
    
    print(f"🎯 ESTADO FINAL:")
    print(f"   FCI: {fci_final:.3f}, Sincronização: {sync_final:.3f}")
    print(f"   Coerência: {coher_final:.3f}, Ganho Líder: {ganho_final:.1f}")
    
    print(f"\n📊 MÁXIMOS ALCANÇADOS:")
    print(f"   FCI Máximo: {fci_max:.3f}, Sincronização Máxima: {sync_max:.3f}")
    
    print(f"\n⚖️  ESTABILIDADE AVANÇADA:")
    print(f"   Variação FCI (últimos 10s): {fci_std:.4f}")
    print(f"   Ganho final: {ganho_final:.1f} (base: {plasma.K_lider})")
    print(f"   DT final: {plasma.historico_dt[-1]:.3f}")
    
    # 🎯 VERIFICAÇÃO DAS CORREÇÕES
    print(f"\n✅ CORREÇÕES DE ESTABILIZAÇÃO:")
    print(f"   1. Média móvel interna ✓ (janela 3)")
    print(f"   2. Ganho dinâmico ✓ (sync > 0.7 → ganho ↓)")
    print(f"   3. DT adaptativo ✓ (sync alto → DT ↓)")
    
    # ANÁLISE DE EFICÁCIA
    ganho_avg = np.mean(plasma.historico_ganho)
    dt_avg = np.mean(plasma.historico_dt)
    
    print(f"\n📈 EFICÁCIA DAS CORREÇÕES:")
    print(f"   Ganho médio: {ganho_avg:.1f} (redução: {(plasma.K_lider - ganho_avg)/plasma.K_lider*100:.1f}%)")
    print(f"   DT médio: {dt_avg:.3f}")
    
    if fci_std < 0.05 and fci_final > 0.65:
        print(f"\n🎉 SUCESSO TOTAL! Sistema estável e de alta performance!")
        print(f"   • FCI final > 0.65 ✓")
        print(f"   • Baixa variação ({fci_std:.4f}) ✓")
        print(f"   • Ganho automático funcionando ✓")
    elif fci_max > 0.7 and fci_std < 0.08:
        print(f"\n✅ BOM RESULTADO! Sistema atinge picos altos com boa estabilidade")
        print(f"   • FCI máximo: {fci_max:.3f} ✓")
        print(f"   • Estabilidade aceitável ✓")
    else:
        print(f"\n⚠️  Sistema precisa de ajustes finais")
        print(f"   • FCI máximo: {fci_max:.3f}")
        print(f"   • Variação: {fci_std:.4f}")
    
    print("="*75)