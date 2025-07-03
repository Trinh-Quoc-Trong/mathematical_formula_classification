# Cách chuyển Repository GitHub thành Public

## Repository hiện tại
Repository: `Trinh-Quoc-Trong/mathematical_formula_classification`
Mục tiêu: Phân lớp 3 ký hiệu toán học

## Phương pháp 1: Thông qua giao diện GitHub Web (Khuyến nghị)

### Bước 1: Truy cập repository
1. Mở trình duyệt và đăng nhập vào GitHub
2. Truy cập: https://github.com/Trinh-Quoc-Trong/mathematical_formula_classification

### Bước 2: Vào Settings
1. Click vào tab **Settings** (ở phía trên cùng của repository)
2. Cuộn xuống phần **Danger Zone** (ở cuối trang)

### Bước 3: Thay đổi visibility
1. Tìm mục **Change repository visibility**
2. Click vào nút **Change visibility**
3. Chọn **Make public**
4. Xác nhận bằng cách:
   - Gõ tên repository: `mathematical_formula_classification`
   - Click **I understand, change repository visibility**

## Phương pháp 2: Sử dụng GitHub CLI (gh)

### Cài đặt GitHub CLI (nếu chưa có)
```bash
# Ubuntu/Debian
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

### Thực hiện lệnh
```bash
# Đăng nhập (nếu chưa đăng nhập)
gh auth login

# Chuyển repository thành public
gh repo edit Trinh-Quoc-Trong/mathematical_formula_classification --visibility public
```

## Phương pháp 3: Sử dụng GitHub API

### Sử dụng curl với Personal Access Token
```bash
curl -X PATCH \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_PERSONAL_ACCESS_TOKEN" \
  https://api.github.com/repos/Trinh-Quoc-Trong/mathematical_formula_classification \
  -d '{"private": false}'
```

## Lưu ý quan trọng

### ⚠️ Cảnh báo trước khi chuyển public:
1. **Kiểm tra thông tin nhạy cảm**: 
   - API keys, passwords, tokens
   - Thông tin cá nhân hoặc dữ liệu nhạy cảm
   - File cấu hình với thông tin bảo mật

2. **Kiểm tra git history**:
   ```bash
   git log --oneline --all
   git show <commit-hash>  # Kiểm tra từng commit
   ```

3. **Làm sạch repository nếu cần**:
   ```bash
   # Xóa file khỏi git history (cẩn thận!)
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch sensitive-file.txt' \
   --prune-empty --tag-name-filter cat -- --all
   ```

### ✅ Lợi ích của repository public:
- Tăng khả năng hiển thị và collaboration
- Có thể được fork và star bởi community
- Tích hợp tốt hơn với các dịch vụ CI/CD miễn phí
- Có thể sử dụng GitHub Pages miễn phí

### 📋 Checklist trước khi public:
- [ ] Đã kiểm tra không có thông tin nhạy cảm
- [ ] README.md đầy đủ thông tin
- [ ] License file (nếu cần)
- [ ] .gitignore phù hợp
- [ ] Code đã được review và clean

## Kiểm tra trạng thái hiện tại

Để kiểm tra trạng thái hiện tại của repository:
```bash
gh repo view Trinh-Quoc-Trong/mathematical_formula_classification --json isPrivate
```

Hoặc truy cập trực tiếp URL: https://github.com/Trinh-Quoc-Trong/mathematical_formula_classification

## Khuyến nghị

**Phương pháp 1 (GitHub Web)** là cách dễ nhất và an toàn nhất cho người mới bắt đầu. Bạn có thể thấy rõ các tùy chọn và xác nhận trước khi thực hiện thay đổi.

Sau khi chuyển thành public, repository sẽ:
- Hiển thị công khai cho mọi người
- Có thể được tìm thấy qua tìm kiếm GitHub
- Cho phép người khác fork và clone
- Tính vào GitHub contribution graph của bạn