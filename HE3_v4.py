import matplotlib.pyplot as plt
import numpy as np

# ── Experimental data ─────────────────────────────────────────────────────────
acc  = [1.2, 0.2, 0.1, 2.2, 0.2, 0.5, 3.8, 0.05, 0.8 ]                # m/s²
vel  = [0.2, 0.3, 0.2, 0.3, 0.2, 0.1, 0.9, 0.5, 1.7]                  # mm/s
disp = [0.002, 0.07, 0.01, 0.003, 0.022, 0.006, 0.014, 0.038, 0.044]  # mm

labels = [
    "60 RPM – Pos 1",   "60 RPM – Pos 2",   "60 RPM – Pos 3",
    "300 RPM – Pos 1",  "300 RPM – Pos 2",  "300 RPM – Pos 3",
    "1080 RPM – Pos 1", "1080 RPM – Pos 2", "1080 RPM – Pos 3",
]
rpm_labels = ['60 RPM', '300 RPM', '1080 RPM']
Z95 = 1.96

# ── Sensor uncertainty ────────────────────────────────────────────────────────
def delta_a(a): return 0.05 * a + 2 * 0.1       # m/s²
def delta_v(v): return 0.05 * v + 2 * 0.1       # mm/s
def delta_d(d): return 0.05 * d + 2 * 0.001     # mm

def freq_ci(f, rel_unc):
    abs_unc = f * rel_unc
    return abs_unc, f - Z95 * abs_unc, f + Z95 * abs_unc

def weighted_geomean(fs, rels):
    ln_fs = np.log(fs)
    ws    = 1.0 / np.array(rels)**2
    ln_fw = np.sum(ws * ln_fs) / np.sum(ws)
    rel_w = 1.0 / np.sqrt(np.sum(ws))
    f_w   = np.exp(ln_fw)
    return f_w, rel_w, ws / np.sum(ws)

def draw_nomogram(ax):
    f_min, f_max, v_min, v_max = 1, 10_000, 0.1, 10_000
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1, 500) 
    ax.set_ylim(v_min, v_max)
    
    # GRID đậm rõ nét theo yêu cầu
    ax.grid(True, which="both", color='#D5D5D5', linestyle='-', linewidth=0.5)
    
    f_space = np.logspace(np.log10(f_min), np.log10(f_max), 300)
    for p in range(-3, 4):
        v_line = 2 * np.pi * f_space * (10**p) / (2 * np.sqrt(2))
        m = (v_line >= v_min) & (v_line <= v_max)
        if m.any(): ax.plot(f_space[m], v_line[m], color='#1f4e79', lw=0.35, alpha=0.15)
    for p in range(-2, 7):
        v_line = ((10**p) / (2 * np.pi * f_space)) * 1e3 * np.sqrt(2)
        m = (v_line >= v_min) & (v_line <= v_max)
        if m.any(): ax.plot(f_space[m], v_line[m], color='#800000', lw=0.35, alpha=0.15)
            
    ax.set_xticks([1,2,3,5,10,20,30,50,100,200,300,500])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Velocity (mm/s) rms', fontsize=11, fontweight='bold', labelpad=8)

# ── Header Output (Console) ───────────────────────────────────────────────────
col = 12
print("\n" + "="*145)
print(f"{'POINT':<22} | {'f_av':>{col}} {'±CI':>{col}} | {'f_vd':>{col}} {'±CI':>{col}} | {'f_ad':>{col}} {'±CI':>{col}} | {'f_WEIGHTED':>{col}} {'±CI':>{col}}")
print("-" * 145)

markers      = ['o', 's', '^']
rpm_colors   = ['#2ca02c', '#ff7f0e', '#9467bd']

# ── Main Loop ─────────────────────────────────────────────────────────────────
for rpm_idx in range(3):
    color = rpm_colors[rpm_idx]
    rpm   = rpm_labels[rpm_idx]

    fig, ax = plt.subplots(figsize=(13, 10))
    draw_nomogram(ax)
    ax.set_title(f'Nomogramme – {rpm}\nTriangles of uncertainty | 95% CI | Weighted centroid', fontsize=11, fontweight='bold', pad=12)

    handles = []

    for pos_idx in range(3):
        i = rpm_idx * 3 + pos_idx
        marker = markers[pos_idx]

        a = acc[i];  v = vel[i];  d = disp[i]
        a_m = a;     v_m = v * 1e-3;  d_m = d * 1e-3

        # Tính toán tần số
        f_av = a_m / (2 * np.pi * v_m)
        f_vd = v_m / (2 * np.pi * d_m)
        f_ad = (1 / (2 * np.pi)) * np.sqrt(a_m / d_m)

        v_av, v_vd = v, v
        v_ad = (a_m / (2 * np.pi * f_ad)) * 1e3 * np.sqrt(2)

        # Tính sai số
        da, dv, dd = delta_a(a), delta_v(v), delta_d(d)
        rel_av, rel_vd, rel_ad = np.sqrt((da/a)**2 + (dv/v)**2), np.sqrt((dv/v)**2 + (dd/d)**2), 0.5 * np.sqrt((da/a)**2 + (dd/d)**2)

        hw_av, lo_av, hi_av = freq_ci(f_av, rel_av)
        hw_vd, lo_vd, hi_vd = freq_ci(f_vd, rel_vd)
        hw_ad, lo_ad, hi_ad = freq_ci(f_ad, rel_ad)

        # Weighted centroid
        f_w, rel_w, _ = weighted_geomean([f_av, f_vd, f_ad], [rel_av, rel_vd, rel_ad])
        hw_w, lo_w, hi_w = freq_ci(f_w, rel_w)
        
        v_w, rel_vw, _ = weighted_geomean([v, v, v_ad], [rel_av, rel_vd, rel_ad])
        _, lo_vw, hi_vw = freq_ci(v_w, rel_vw)

        # XUẤT KẾT QUẢ RA CONSOLE (Tất cả RPM đều dùng chung định dạng 2 chữ số thập phân)
        print(f"{labels[i]:<22} | {f_av:>{col}.2f} {hw_av:>{col}.2f} | {f_vd:>{col}.2f} {hw_vd:>{col}.2f} | {f_ad:>{col}.2f} {hw_ad:>{col}.2f} | {f_w:>{col}.2f} {hw_w:>{col}.2f}")

        # Vẽ Triangle & Markers (Giữ nguyên các thiết kế mảnh/mờ bạn đã chọn)
        ax.fill([f_av, f_vd, f_ad, f_av], [v_av, v_vd, v_ad, v_av], color=color, alpha=0.05, zorder=3)
        ax.plot([f_av, f_vd, f_ad, f_av], [v_av, v_vd, v_ad, v_av], color=color, linestyle=[':', '--', '-.'][pos_idx], lw=1.4, zorder=4)
        ax.scatter([f_av, f_vd, f_ad], [v_av, v_vd, v_ad], color=color, marker=marker, s=45, edgecolor='black', lw=0.8, zorder=8)

        # Centroid ★ và CI mảnh
        ax.errorbar(f_w, v_w, xerr=[[max(f_w - lo_w, 0)], [max(hi_w - f_w, 0)]], yerr=[[max(v_w - lo_vw, 0)], [max(hi_vw - v_w, 0)]],
                    fmt='none', color=color, elinewidth=1.2, capsize=3, capthick=1.2, zorder=10, alpha=0.95)
        ax.scatter(f_w, v_w, color=color, marker='*', s=170, edgecolor='black', lw=1.0, zorder=11)

        h = ax.scatter([], [], color=color, marker=marker, s=55, edgecolor='black', lw=0.8,
                       label=f'Pos {pos_idx+1}  (f_w={f_w:.1f} Hz, v={v:.2f} mm/s)')
        handles.append(h)

    # Thêm Legend
    handles.append(ax.scatter([], [], color=color, marker='*', s=150, edgecolor='black', lw=1.0, label='Weighted centroid ★'))
    handles.append(ax.errorbar([], [], xerr=1, fmt='none', color='gray', elinewidth=1.2, capsize=3, capthick=1.2, label='95% CI'))
    ax.legend(handles=handles, loc='upper left', fontsize=9, framealpha=0.95, facecolor='white', edgecolor='gray')
    plt.tight_layout()
    if rpm_idx < 2: print("-" * 145) # Dòng kẻ phân cách giữa các RPM ở output

print("="*145 + "\n")
plt.show()