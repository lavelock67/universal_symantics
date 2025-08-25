#!/bin/bash

# NSM API Production Deployment Script
# This script deploys the enhanced NSM API with all advanced components

set -e

echo "🚀 Starting NSM API Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Create SSL certificates for development
create_ssl_certs() {
    print_status "Creating SSL certificates for development..."
    
    if [ ! -d "ssl" ]; then
        mkdir -p ssl
    fi
    
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_success "SSL certificates created"
    else
        print_status "SSL certificates already exist"
    fi
}

# Create Grafana configuration directories
setup_grafana() {
    print_status "Setting up Grafana configuration..."
    
    mkdir -p grafana/dashboards
    mkdir -p grafana/datasources
    
    # Create default datasource
    cat > grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    print_success "Grafana configuration created"
}

# Build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    # Stop any existing services
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose up -d --build
    
    print_success "Services deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for NSM API
    print_status "Waiting for NSM API..."
    timeout=120
    counter=0
    while ! curl -f http://localhost:8001/health &> /dev/null; do
        sleep 2
        counter=$((counter + 2))
        if [ $counter -ge $timeout ]; then
            print_error "NSM API failed to start within $timeout seconds"
            exit 1
        fi
    done
    
    # Wait for Prometheus
    print_status "Waiting for Prometheus..."
    counter=0
    while ! curl -f http://localhost:9090/-/healthy &> /dev/null; do
        sleep 2
        counter=$((counter + 2))
        if [ $counter -ge $timeout ]; then
            print_warning "Prometheus failed to start within $timeout seconds"
            break
        fi
    done
    
    # Wait for Grafana
    print_status "Waiting for Grafana..."
    counter=0
    while ! curl -f http://localhost:3000/api/health &> /dev/null; do
        sleep 2
        counter=$((counter + 2))
        if [ $counter -ge $timeout ]; then
            print_warning "Grafana failed to start within $timeout seconds"
            break
        fi
    done
    
    print_success "All services are ready"
}

# Display deployment information
show_deployment_info() {
    print_success "🎉 Deployment completed successfully!"
    echo
    echo "📊 Service URLs:"
    echo "  NSM API:        http://localhost:8001"
    echo "  NSM API (HTTPS): https://localhost"
    echo "  Prometheus:     http://localhost:9090"
    echo "  Grafana:        http://localhost:3000 (admin/admin)"
    echo
    echo "🔧 API Endpoints:"
    echo "  Health Check:   GET /health"
    echo "  Metrics:        GET /metrics"
    echo "  Primes List:    GET /primes"
    echo "  Detection:      POST /detect"
    echo "  DeepNSM:        POST /deepnsm"
    echo "  MDL Validation: POST /mdl"
    echo "  Temporal:       POST /temporal"
    echo "  NSM Generation: POST /generate/nsm"
    echo "  Grammar Gen:    POST /generate/grammar"
    echo "  Risk Router:    POST /router/route"
    echo "  Router Stats:   GET /router/stats"
    echo
    echo "📈 Monitoring:"
    echo "  Prometheus metrics available at /metrics"
    echo "  Grafana dashboards for visualization"
    echo
    echo "🛠️  Management Commands:"
    echo "  View logs:      docker-compose logs -f"
    echo "  Stop services:  docker-compose down"
    echo "  Restart:        docker-compose restart"
    echo "  Update:         git pull && docker-compose up -d --build"
}

# Test the deployment
test_deployment() {
    print_status "Testing deployment..."
    
    # Test health endpoint
    if curl -f http://localhost:8001/health &> /dev/null; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        exit 1
    fi
    
    # Test metrics endpoint
    if curl -f http://localhost:8001/metrics &> /dev/null; then
        print_success "Metrics endpoint working"
    else
        print_warning "Metrics endpoint not working"
    fi
    
    # Test basic detection
    response=$(curl -s -X POST http://localhost:8001/detect \
        -H "Content-Type: application/json" \
        -d '{"text": "I think you know the truth", "language": "en"}')
    
    if echo "$response" | grep -q "detected_primes"; then
        print_success "Detection endpoint working"
    else
        print_warning "Detection endpoint may have issues"
    fi
    
    print_success "Deployment test completed"
}

# Main deployment process
main() {
    echo "🎯 NSM API Production Deployment"
    echo "================================="
    echo
    
    check_dependencies
    create_ssl_certs
    setup_grafana
    deploy_services
    wait_for_services
    test_deployment
    show_deployment_info
}

# Run main function
main "$@"
