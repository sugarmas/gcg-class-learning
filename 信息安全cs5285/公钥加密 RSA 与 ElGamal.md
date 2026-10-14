# 公钥加密 RSA 与 ElGamal

Lecture 4 的主线：非对称密码为什么被发明、建立在哪两类数学难题上、以及 RSA / ElGamal / Diffie-Hellman 各自「是以什么为代价换来的安全」。

## 对称 vs 非对称

对称密钥的规模化问题：n 方系统要 $n(n-1)/2$ 个密钥、每人维护 n−1 个。两条出路：
1. **集中式密钥管理**——每人与服务器共享一个长期密钥，通信时由服务器发 **session key**（真正数据用 session key 加密）。只需 n 个长期密钥，但服务器是**单点故障与性能瓶颈**。
2. **公钥密码**——接收方持 key pair，公钥公开，任何人可加密，只有私钥持有者能解密。

| | 对称 | 非对称 |
|---|---|---|
| 信任 | 双方须互信 | 无需互信 |
| 密钥 | 同一 key | 公钥 / 私钥成对 |
| 攻击 | 穷举 | 解数学难题（factorization / DLP） |
| 速度 | 快 | 慢 100–1000 倍 |
| key | 小 | 大 |
| 例 | DES/3DES/AES | RSA/ElGamal/ECC |

现代系统极少用 PKE 直接加密数据，更常见的是用它**封装 session key**。

## 单向函数（不要与哈希混淆）

**单向函数**：给定输入算输出容易，给定输出反推输入困难。公钥密码建立在数学单向（陷门）函数上，候选两族：**因式分解**与**离散对数**。（哈希也单向，但它是压缩到定长、无 key；这里的「单向函数」指构造加解密用的数学函数。）

## RSA

构造：
1. 随机取两个等长大素数 $p,q$（如各 512 bit）；
2. $n=pq$（public modulus）；
3. 取 $e$ 使 $\gcd(e,\varphi(n))=1$，其中 $\varphi(n)=(p-1)(q-1)$；
4. $d=e^{-1}\bmod\varphi(n)$（扩展欧几里得）；
5. 公钥 $(n,e)$，私钥 $d$；
6. 加密 $C=M^e\bmod n$，解密 $M=C^d\bmod n$。

**明文必须 $M<n$**（这就是「块大小」上限）。例：$p=47,q=71\Rightarrow n=3337,\varphi=3220$，$e=79\Rightarrow d=1019$；$688\to1570\to688$。计算私钥依赖扩展欧几里得与 Euler totient。

**安全性 = 因式分解困难**。攻击者若能分解 $n=pq$，就得到 $\varphi(n)$，再由 $e$ 求 $d$，体制即破。**RSA 问题（RSAP）**：给 $n=pq$、$e$、$c$，求 $m$ 使 $m^e\equiv c\pmod n$。已知「能分解 ⇒ 能解 RSAP」，反向是否成立未证明但普遍相信。对 RSA 做**暴力 key 搜索没有意义**（key 太大），要打就打分解。

**key 长度与强度**（RSA 强度增长很慢）：RSA-1024≈80-bit 对称、RSA-2048≈112-bit、RSA-3072≈128-bit、RSA-15360≈AES-256。签名耗时约与 key 长度**立方**成正比（RSA-2048 约 8× RSA-1024）。历史上 512-bit 已被淘汰，当前 2048 bit 广泛使用、3072 bit 渐成推荐。

## ElGamal

素 $p$，$g\in\mathbb Z_p^*$ 为生成元。私钥 $x\in\mathbb Z_{p-1}$，公钥 $y=g^x\bmod p$。
加密：随机取 $r$，$A=g^r\bmod p$，$B=M\,y^r\bmod p$，密文 $C=(A,B)$。
解密：$K=A^x\bmod p$，$M=B\,K^{-1}\bmod p$（需模逆）。

- **概率加密**：同一明文两次加密密文不同（$r$ 不同）；RSA 是确定性的，相同明文得相同密文。这是 ElGamal 的结构性优势。
- **安全性 = 离散对数困难**。破法三条路：由 $A$ 求 $r$、由 $y$ 求 $x$（都是 DLP），或由 $A,y$ 求 $g^{rx}$（DH 问题）。

## Diffie-Hellman 密钥交换

**它是密钥交换，不是加密**（不能用来传消息）。Alice 取 $a$ 发 $g^a$，Bob 取 $b$ 发 $g^b$，双方各自算出 $g^{ab}$ 作为对称 key。
对**被动**窃听者安全（依赖 DH 问题困难）。对**主动**攻击者不安全：**man-in-the-middle**——Trudy 与 Alice 共享 $g^{at}$、与 Bob 共享 $g^{bt}$，双方都不知 Trudy 存在。要用认证（签名、证书）才能防。

## 后量子密码

Shor 算法可高效解因式分解与离散对数 → RSA / ElGamal / DH 全被量子计算击穿；**对称密码不受影响**。NIST 已选 ML-KEM(Kyber)、ML-DSA(Dilithium)、FN-DSA(Falcon)、SLH-DSA(SPHINCS+)，主要基于 lattice / code-based 的 NP-hard 问题。

## 相关

- [[密码学数论基础]]
- [[哈希 消息认证码与数字签名]]
- [[CS5285 Index]]
